"""
Training Worker — runs in a subprocess, isolated from the main server.

Communicates via JSONL on stdout:
  {"event": "status", "phase": "loading_model"}
  {"event": "progress", "step": 1, "total_steps": 100, "loss": 1.23, "lr": 1e-4, "epoch": 0.5, "eta_seconds": 300}
  {"event": "metrics", "eval_loss": 1.1, "grad_norm": 0.05}
  {"event": "complete", "output_dir": "...", "final_loss": 0.89}
  {"event": "error", "message": "CUDA out of memory"}

Checks for stop signal by polling a stop file (path sent in config).
"""

import json
import os
import sys
import time
import math
from pathlib import Path


def emit(event_type: str, **kwargs):
    """Emit a JSON event to stdout for the parent process to consume."""
    data = {"event": event_type, **kwargs}
    print(json.dumps(data), flush=True)


def check_stop(stop_file: str) -> bool:
    """Check if the stop signal file exists."""
    return os.path.exists(stop_file)


def run(config: dict):
    """Main training entry point. Called after imports to avoid import latency in event emission."""
    niche = config.get("niche", "default")
    data_dir = config.get("data_dir")
    adapter_path = config.get("adapter_path", f"models/adapters/{niche}")
    base_model = config.get("base_model", "mlx-community/Qwen2.5-7B-Instruct-4bit")
    stop_file = config.get("stop_file")

    # Training hyperparams
    lora_rank = config.get("lora_rank", 16)
    lora_alpha = config.get("lora_alpha", 32)
    learning_rate = config.get("learning_rate", 1e-4)
    batch_size = config.get("batch_size", 4)
    epochs = config.get("epochs", 3)
    max_seq_length = config.get("max_seq_length", 2048)
    grad_checkpoint = config.get("grad_checkpoint", True)
    resume_adapter = config.get("resume_adapter", "")

    # Dataset settings
    dataset_type = config.get("dataset_type", "local")  # "local" or "bigset"
    dataset_desc = config.get("dataset_desc", "")
    max_rows = config.get("max_rows", 50)

    emit("status", phase="initializing", message="Setting up training environment")

    # Lazy import MLX to keep startup fast
    emit("status", phase="loading_model", message=f"Loading base model: {base_model}")

    if check_stop(stop_file):
        emit("status", phase="stopped", message="Training cancelled before start")
        return

    # Import here so the process can report progress before heavy imports
    emit("status", phase="importing_modules", message="Importing MLX modules...")

    try:
        from mlx_lm import load, generate
        from mlx_lm.lora import TrainingArgs, train_model, load_dataset
        import mlx.core as mx
        import mlx.optimizers as optim
    except ImportError as e:
        emit("error", message=f"Failed to import MLX: {e}")
        return

    emit("status", phase="configuring", message="Preparing training configuration")

    # Check for stop after imports
    if check_stop(stop_file):
        emit("status", phase="stopped", message="Training cancelled during setup")
        return

    # --- DATA PREP ---
    emit("status", phase="preparing_data", message="Preparing training data")

    # Count training examples
    train_path = os.path.join(data_dir, "train.jsonl")
    valid_path = os.path.join(data_dir, "valid.jsonl")

    if not os.path.exists(train_path):
        emit("error", message=f"Training data not found: {train_path}")
        return

    with open(train_path) as f:
        num_train = sum(1 for _ in f if _.strip())

    with open(valid_path) as f:
        num_valid = sum(1 for _ in f if _.strip())

    if num_train == 0:
        emit("error", message="Training data is empty")
        return

    emit("status", phase="data_ready", message=f"Data ready: {num_train} train, {num_valid} validation")

    # --- CREATE ARGS ---
    # Build a SimpleNamespace-style args object for mlx_lm.lora
    class Args:
        pass

    args = Args()
    args.model = base_model
    args.train = True
    args.test = False
    args.fine_tune_type = "lora"
    args.optimizer = "adamw"
    args.optimizer_config = {"adamw": {}}
    args.mask_prompt = False
    args.num_layers = -1
    args.batch_size = batch_size
    args.val_batches = min(25, max(1, num_valid))
    args.learning_rate = learning_rate
    args.steps_per_report = max(1, num_train // batch_size // 10)
    args.steps_per_eval = max(1, num_train // batch_size // 5)
    args.save_every = max(1, num_train // batch_size)
    args.max_seq_length = max_seq_length
    args.grad_checkpoint = grad_checkpoint
    args.grad_accumulation_steps = 1
    args.lr_schedule = None
    args.report_to = None
    args.project_name = niche
    args.seed = 42
    args.resume_adapter_file = None
    args.data = data_dir
    args.clear_cache_threshold = 0

    lora_scale = lora_alpha / lora_rank
    args.lora_parameters = {
        "rank": lora_rank,
        "dropout": 0.0,
        "scale": lora_scale,
    }

    steps_per_epoch = max(1, num_train // batch_size)
    total_iters = steps_per_epoch * epochs

    emit("status", phase="training",
         message=f"Starting training: {epochs} epochs, {total_iters} iterations",
         total_steps=total_iters)

    # --- LOAD MODEL ---
    try:
        if check_stop(stop_file):
            emit("status", phase="stopped", message="Training cancelled")
            return

        emit("status", phase="loading_model_weights", message="Loading model weights...")
        model, tokenizer = load(base_model, tokenizer_config={"trust_remote_code": True})
        num_params = sum(p.size for p in model.parameters() if hasattr(p, 'size'))
        emit("status", phase="model_loaded",
             message=f"Model loaded: {num_params/1e9:.1f}B parameters")
    except Exception as e:
        emit("error", message=f"Failed to load model: {e}")
        return

    # --- LOAD DATASET ---
    if check_stop(stop_file):
        emit("status", phase="stopped", message="Training cancelled")
        return

    try:
        train_set, valid_set, _ = load_dataset(args, tokenizer)
        emit("status", phase="dataset_loaded",
             message=f"Dataset loaded: {len(train_set)} training samples")
    except Exception as e:
        emit("error", message=f"Failed to load dataset: {e}")
        return

    # --- TRAIN ---
    # We override train_model to capture progress callbacks
    emit("status", phase="training_started", message="Starting training loop")

    # Set up seed
    mx.random.seed(args.seed)

    # Freeze and apply LoRA
    model.freeze()
    from mlx_lm.tuner.utils import linear_to_lora_layers

    try:
        linear_to_lora_layers(model, args.num_layers, args.lora_parameters, use_dora=False)
    except Exception as e:
        emit("error", message=f"Failed to apply LoRA: {e}")
        return

    # Continue training from a previous adapter (incremental fine-tuning): load the
    # prior run's LoRA weights into the freshly-applied layers so optimization picks
    # up where it left off instead of from random init. The rank/alpha here must match
    # the source adapter (the caller is responsible for that). strict=False tolerates
    # the base model's frozen (non-adapter) params being absent from the file.
    if resume_adapter:
        resume_file = resume_adapter
        if os.path.isdir(resume_file):
            resume_file = os.path.join(resume_file, "adapters.safetensors")
        if os.path.exists(resume_file):
            try:
                model.load_weights(resume_file, strict=False)
                args.resume_adapter_file = resume_file
                emit("status", phase="resumed_adapter",
                     message=f"Resumed from adapter: {resume_file}")
            except Exception as e:
                emit("error", message=f"Failed to load resume adapter '{resume_file}': {e}")
                return
        else:
            emit("error", message=f"Resume adapter not found: {resume_file}")
            return

    # Count trainable parameters (handle quantized models gracefully)
    try:
        trainable = sum(p.size for p in model.parameters() if hasattr(p, 'size') and not p.isfrozen())
        total = sum(p.size for p in model.parameters() if hasattr(p, 'size'))
        if total > 0:
            pct = 100 * trainable / total
            emit("status", phase="lora_configured",
                 message=f"Trainable params: {trainable:,} / {total:,} ({pct:.2f}%)")
        else:
            # Quantized model — count layers instead
            t = sum(1 for p in model.parameters() if hasattr(p, 'size') and not p.isfrozen())
            tt = sum(1 for p in model.parameters() if hasattr(p, 'size'))
            emit("status", phase="lora_configured",
                 message=f"LoRA applied: {t} trainable layers / {tt} total")
    except Exception:
        emit("status", phase="lora_configured",
             message="LoRA configured (quantized model)")

    # Setup adapter path
    from pathlib import Path as PPath
    adapter_path_obj = PPath(adapter_path)
    adapter_path_obj.mkdir(parents=True, exist_ok=True)
    adapter_file = adapter_path_obj / "adapters.safetensors"

    # Save config
    from mlx_lm.utils import save_config
    save_config(
        {
            "model": base_model,
            "fine_tune_type": "lora",
            # num_layers is required by mlx_lm.tuner.utils.load_adapters at
            # merge/export time (it rebuilds the LoRA layers from this config).
            # -1 == apply LoRA to all layers, matching how we trained.
            "num_layers": args.num_layers,
            "lora_parameters": args.lora_parameters,
        },
        str(adapter_path_obj / "adapter_config.json"),
    )

    # Build training args — val_batches must not exceed available validation data
    if num_valid >= batch_size:
        val_batches = max(1, num_valid // batch_size)
    else:
        val_batches = 0  # Skip validation if too few examples
    training_args = TrainingArgs(
        batch_size=batch_size,
        iters=total_iters,
        val_batches=val_batches,
        steps_per_report=1,  # Report every step for progress
        steps_per_eval=args.steps_per_eval,
        steps_per_save=args.save_every,
        adapter_file=adapter_file,
        max_seq_length=max_seq_length,
        grad_checkpoint=grad_checkpoint,
        grad_accumulation_steps=1,
    )

    # Create optimizer
    from mlx_lm.tuner.trainer import train, evaluate

    opt = optim.AdamW(learning_rate=learning_rate)

    from mlx_lm.tuner.datasets import CacheDataset

    train_dataset = CacheDataset(train_set)
    val_dataset = CacheDataset(valid_set)

    # Custom training loop that emits progress via the correct callback interface
    start_time = time.time()
    global_step = 0
    best_loss = float("inf")

    from mlx_lm.tuner.callbacks import TrainingCallback

    class ProgressCallback(TrainingCallback):
        """Uses mlx_lm's built-in callback interface. Checks stop_file periodically."""

        def __init__(self, stop_file: str):
            self._last_check = 0
            self.stop_file = stop_file
            self.stop_requested = False

        def on_train_loss_report(self, train_info: dict):
            nonlocal global_step, best_loss, start_time

            global_step += 1
            loss_val = float(train_info.get("train_loss", 0))
            lr_val = train_info.get("learning_rate", 0)
            tokens_sec = train_info.get("tokens_per_second", 0)

            if loss_val < best_loss:
                best_loss = loss_val

            elapsed = time.time() - start_time
            progress = global_step / total_iters if total_iters > 0 else 0
            eta = elapsed / progress - elapsed if progress > 0 else 0
            epoch = (global_step / steps_per_epoch) if steps_per_epoch > 0 else 0

            emit("progress",
                 step=global_step,
                 total_steps=total_iters,
                 loss=round(loss_val, 6),
                 lr=round(lr_val, 10) if lr_val else 0,
                 progress_percent=round(progress * 100, 1),
                 epoch=round(epoch, 2),
                 elapsed_seconds=round(elapsed),
                 eta_seconds=round(eta),
                 tokens_per_sec=round(tokens_sec, 1) if tokens_sec else 0,
            )

            # Check for stop signal periodically
            self._last_check += 1
            if self._last_check >= 3:
                self._last_check = 0
                if os.path.exists(self.stop_file):
                    self.stop_requested = True

        def on_val_loss_report(self, val_info: dict):
            pass

    callback = ProgressCallback(stop_file)

    try:
        train(
            model=model,
            args=training_args,
            optimizer=opt,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            training_callback=callback,
        )
    except Exception as e:
        if callback.stop_requested:
            emit("status", phase="stopped", message="Training stopped by user")
        else:
            emit("error", message=f"Training failed: {e}")
        # Save partial adapter on any exit
        try:
            model.save_weights(str(adapter_file))
            emit("checkpoint_saved", path=str(adapter_file))
        except Exception:
            pass
        return

    # --- SAVE ---
    emit("status", phase="saving", message="Saving model adapter...")
    model.save_weights(str(adapter_file))
    emit("checkpoint_saved", path=str(adapter_file))

    # --- SAVE METADATA ---
    metadata = {
        "niche": niche,
        "base_model": base_model,
        "epochs": epochs,
        "batch_size": batch_size,
        "lora_rank": lora_rank,
        "learning_rate": learning_rate,
        "training_examples": num_train,
        "total_iters": global_step,
        "final_loss": round(best_loss, 6),
        "duration_seconds": round(time.time() - start_time),
    }
    with open(os.path.join(adapter_path_obj, "training_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    emit("complete",
         output_dir=adapter_path,
         final_loss=round(best_loss, 6),
         total_steps=global_step,
         duration_seconds=round(time.time() - start_time),
         num_train_examples=num_train)


if __name__ == "__main__":
    # Read config from stdin (first line should be JSON)
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
