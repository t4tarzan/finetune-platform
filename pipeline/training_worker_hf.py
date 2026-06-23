"""
Training Worker (HuggingFace / CPU) — Linux drop-in for training_worker.py.

The MLX worker (training_worker.py) only runs on Apple Silicon. This worker
provides the same stdin-config / stdout-JSONL-event contract using
PyTorch + transformers + PEFT, so it runs on Linux CPU (no Metal, no CUDA).

Communicates via JSONL on stdout (identical event protocol to the MLX worker):
  {"event": "status", "phase": "loading_model", "message": "..."}
  {"event": "progress", "step": 1, "total_steps": 100, "loss": 1.23, "lr": 1e-4, "epoch": 0.5, ...}
  {"event": "complete", "output_dir": "...", "final_loss": 0.89}
  {"event": "error", "message": "..."}

Checks for stop signal by polling a stop file (path sent in config).
"""

import json
import os
import sys
import time
from pathlib import Path

# CPU default — config's default base_model is an MLX 4-bit 7B id that HF cannot
# load and that is too large to train on 4 CPU cores anyway.
CPU_DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
# Hard cap on sequence length on CPU (config default of 2048 is wasteful here).
CPU_MAX_SEQ_LEN = 1024


def emit(event_type: str, **kwargs):
    """Emit a JSON event to stdout for the parent process to consume."""
    print(json.dumps({"event": event_type, **kwargs}), flush=True)


def check_stop(stop_file: str) -> bool:
    return bool(stop_file) and os.path.exists(stop_file)


def resolve_model_id(base_model: str) -> str:
    """Map non-loadable ids (MLX-community, Ollama tags, empty) to a CPU HF id.

    A real HF repo id (contains '/', no ':', not mlx-community) is passed through.
    """
    if not base_model:
        return CPU_DEFAULT_MODEL
    if "mlx-community" in base_model or ":" in base_model or "/" not in base_model:
        return CPU_DEFAULT_MODEL
    return base_model


def load_jsonl(path: str):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def run(config: dict):
    niche = config.get("niche", "default")
    data_dir = config.get("data_dir")
    adapter_path = config.get("adapter_path", f"models/adapters/{niche}")
    requested_model = config.get("base_model", "")
    stop_file = config.get("stop_file")

    lora_rank = int(config.get("lora_rank", 16))
    lora_alpha = int(config.get("lora_alpha", 32))
    learning_rate = float(config.get("learning_rate", 1e-4))
    batch_size = int(config.get("batch_size", 4))
    epochs = float(config.get("epochs", 3))
    max_seq_length = min(int(config.get("max_seq_length", 2048)), CPU_MAX_SEQ_LEN)

    base_model = resolve_model_id(requested_model)

    # Incremental retraining: continue from a previously trained adapter instead of
    # a fresh LoRA. We load it onto the exact base it was trained on (recorded in
    # base_model.txt) so the weights line up.
    resume_adapter = config.get("resume_adapter", "") or ""
    resume_valid = bool(resume_adapter) and os.path.exists(
        os.path.join(resume_adapter, "adapter_config.json"))
    if resume_valid:
        marker = os.path.join(resume_adapter, "base_model.txt")
        if os.path.exists(marker):
            recorded = open(marker).read().strip()
            if recorded:
                base_model = recorded

    emit("status", phase="initializing",
         message=f"CPU training. base_model={base_model} (requested: {requested_model or 'default'})"
                 + (f", resuming from {resume_adapter}" if resume_valid else ""))

    if check_stop(stop_file):
        emit("status", phase="stopped", message="Training cancelled before start")
        return

    # --- DATA ---
    emit("status", phase="preparing_data", message="Loading training data")
    train_path = os.path.join(data_dir, "train.jsonl")
    valid_path = os.path.join(data_dir, "valid.jsonl")
    train_rows = load_jsonl(train_path)
    valid_rows = load_jsonl(valid_path)
    num_train = len(train_rows)
    num_valid = len(valid_rows)
    if num_train == 0:
        emit("error", message=f"Training data not found or empty: {train_path}")
        return
    emit("status", phase="data_ready",
         message=f"Data ready: {num_train} train, {num_valid} validation")

    # --- HEAVY IMPORTS (after first events so the UI shows progress) ---
    emit("status", phase="importing_modules", message="Importing PyTorch / transformers / PEFT...")
    try:
        import torch
        torch.set_num_threads(max(1, os.cpu_count() or 1))
        from datasets import Dataset
        import transformers
        from transformers import (
            AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments,
            DataCollatorForLanguageModeling, TrainerCallback,
        )
        from peft import LoraConfig, get_peft_model, PeftModel
        # Keep stdout clean: only our JSON events should reach the manager.
        transformers.logging.set_verbosity_error()
        transformers.logging.disable_progress_bar()
    except Exception as e:
        emit("error", message=f"Failed to import training libraries: {e}")
        return

    # --- LOAD MODEL + TOKENIZER ---
    emit("status", phase="loading_model", message=f"Loading base model: {base_model}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            base_model, dtype=torch.float32, trust_remote_code=True,
        )
        model.config.use_cache = False
    except Exception as e:
        emit("error", message=f"Failed to load model: {e}")
        return

    if check_stop(stop_file):
        emit("status", phase="stopped", message="Training cancelled during setup")
        return

    # --- TOKENIZE (chat template: prompt->user, completion->assistant) ---
    emit("status", phase="tokenizing", message="Applying chat template and tokenizing")

    def to_text(row):
        messages = [
            {"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": row["completion"]},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False)

    def tokenize(batch_texts):
        out = tokenizer(batch_texts, truncation=True, max_length=max_seq_length)
        return out

    try:
        train_texts = [to_text(r) for r in train_rows]
        train_ds = Dataset.from_dict(tokenize(train_texts))
    except Exception as e:
        emit("error", message=f"Failed to tokenize training data: {e}")
        return

    # --- LoRA ---
    emit("status", phase="configuring", message="Applying LoRA adapter")
    try:
        if resume_valid:
            # Load the previous adapter as trainable and keep fine-tuning it.
            model = PeftModel.from_pretrained(model, resume_adapter, is_trainable=True)
            emit("status", phase="lora_resumed",
                 message=f"Continuing training from adapter: {resume_adapter}")
        else:
            peft_config = LoraConfig(
                r=lora_rank, lora_alpha=lora_alpha, lora_dropout=0.0,
                target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM", bias="none",
            )
            model = get_peft_model(model, peft_config)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        emit("status", phase="lora_configured",
             message=f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    except Exception as e:
        emit("error", message=f"Failed to apply LoRA: {e}")
        return

    steps_per_epoch = max(1, num_train // batch_size)
    total_iters = max(1, int(steps_per_epoch * epochs))
    emit("status", phase="training", message=f"Starting training: {epochs} epochs, ~{total_iters} steps",
         total_steps=total_iters)

    # --- PROGRESS / STOP CALLBACK ---
    start_time = time.time()
    state_holder = {"best_loss": float("inf"), "stopped": False}

    class ProgressCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kw):
            logs = logs or {}
            if "loss" not in logs:
                return
            loss_val = float(logs["loss"])
            if loss_val < state_holder["best_loss"]:
                state_holder["best_loss"] = loss_val
            step = state.global_step
            elapsed = time.time() - start_time
            progress = step / total_iters if total_iters else 0
            eta = (elapsed / progress - elapsed) if progress > 0 else 0
            emit("progress",
                 step=step, total_steps=total_iters,
                 loss=round(loss_val, 6),
                 lr=round(float(logs.get("learning_rate", 0) or 0), 10),
                 progress_percent=round(progress * 100, 1),
                 epoch=round(float(state.epoch or 0), 2),
                 elapsed_seconds=round(elapsed),
                 eta_seconds=round(eta),
                 tokens_per_sec=0,
                 grad_norm=round(float(logs["grad_norm"]), 6) if logs.get("grad_norm") is not None else None)

        def on_step_end(self, args, state, control, **kw):
            if check_stop(stop_file):
                state_holder["stopped"] = True
                control.should_training_stop = True
            return control

    # --- TRAIN ---
    out_dir = os.path.join("/tmp", f"hf_trainer_{niche}")
    targs = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        logging_steps=1,
        save_strategy="no",
        eval_strategy="no",
        report_to=[],
        use_cpu=True,
        dataloader_num_workers=0,
        disable_tqdm=True,
        log_level="error",
        seed=42,
    )
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(
        model=model, args=targs, train_dataset=train_ds,
        data_collator=collator, callbacks=[ProgressCallback()],
    )

    emit("status", phase="training_started", message="Starting training loop")
    try:
        trainer.train()
    except Exception as e:
        if state_holder["stopped"]:
            emit("status", phase="stopped", message="Training stopped by user")
        else:
            emit("error", message=f"Training failed: {e}")
        if _save_adapter(model, tokenizer, adapter_path, base_model):
            emit("checkpoint_saved", path=adapter_path)
        return

    # --- SAVE ---
    emit("status", phase="saving", message="Saving LoRA adapter...")
    if not _save_adapter(model, tokenizer, adapter_path, base_model):
        emit("error", message=f"Failed to save LoRA adapter to {adapter_path}")
        return
    emit("checkpoint_saved", path=adapter_path)

    final_loss = round(state_holder["best_loss"], 6) if state_holder["best_loss"] != float("inf") else None
    metadata = {
        "niche": niche, "base_model": base_model, "epochs": epochs,
        "batch_size": batch_size, "lora_rank": lora_rank, "learning_rate": learning_rate,
        "training_examples": num_train, "total_iters": trainer.state.global_step,
        "final_loss": final_loss, "duration_seconds": round(time.time() - start_time),
        "backend": "huggingface-cpu",
    }
    with open(os.path.join(adapter_path, "training_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    if state_holder["stopped"]:
        emit("status", phase="stopped", message="Training stopped by user (checkpoint saved)")
        return

    emit("complete",
         output_dir=adapter_path, final_loss=final_loss,
         total_steps=trainer.state.global_step,
         duration_seconds=round(time.time() - start_time),
         num_train_examples=num_train)


def _save_adapter(model, tokenizer, adapter_path: str, base_model: str) -> bool:
    """Save the LoRA adapter. Returns True on success, False on failure (surfaced
    by the caller) — a swallowed failure here would let training report success
    with no adapter on disk, and the export step would silently serve the base model."""
    Path(adapter_path).mkdir(parents=True, exist_ok=True)
    try:
        model.save_pretrained(adapter_path)          # adapter_model.safetensors + adapter_config.json
        tokenizer.save_pretrained(adapter_path)
        # Record the resolved base model so the HF exporter knows what to merge into.
        with open(os.path.join(adapter_path, "base_model.txt"), "w") as f:
            f.write(base_model)
        return True
    except Exception as e:
        print(f"adapter save failed: {e}", file=sys.stderr, flush=True)
        return False


if __name__ == "__main__":
    config_line = sys.stdin.readline().strip()
    if not config_line:
        print(json.dumps({"event": "error", "message": "No config received"}), flush=True)
        sys.exit(1)
    try:
        config = json.loads(config_line)
    except json.JSONDecodeError as e:
        print(json.dumps({"event": "error", "message": f"Invalid config JSON: {e}"}), flush=True)
        sys.exit(1)
    run(config)
