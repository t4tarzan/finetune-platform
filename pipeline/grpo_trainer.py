"""
GRPO Reinforcement Learning Trainer — subprocess worker.

Implements Group Relative Policy Optimization (GRPO) with:
  - Binary reward (1 = correct, 0 = incorrect) from an LLM judge
  - Group-based advantage normalization
  - Works on top of a LoRA-fine-tuned MLX model

Reference: https://arxiv.org/abs/2402.03300 (DeepSeekMath)
"""

import json
import os
import sys
import time
import math
from pathlib import Path
from typing import Optional

# MLX is Apple-Silicon only. Guard the import so this subprocess fails gracefully
# with a clear event on Linux instead of an ImportError traceback. (GRPO has no
# CPU/HF backend yet — see mlx_available() dispatch in training_manager.)
try:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
except ImportError:
    mx = nn = optim = None


def emit(event_type: str, **kwargs):
    data = {"event": event_type, **kwargs}
    print(json.dumps(data), flush=True)


def check_stop(stop_file: str) -> bool:
    return os.path.exists(stop_file)


def run(config: dict):
    """Main GRPO training entry point."""
    if mx is None:
        emit("error", message="GRPO requires MLX (Apple Silicon) and is not available on this platform.")
        return

    niche = config.get("niche", "default")
    adapter_path = config.get("adapter_path", f"models/adapters/{niche}")
    base_model = config.get("base_model", "mlx-community/Qwen2.5-7B-Instruct-4bit")
    stop_file = config.get("stop_file")
    data_path = config.get("data_path", "data/grpo_train.jsonl")
    judge_model = config.get("judge_model", "claude-sonnet-4-6")  # from commandcode API

    # GRPO hyperparams
    lr = config.get("learning_rate", 5e-6)
    group_size = config.get("group_size", 4)  # Completions per prompt
    max_prompt_length = config.get("max_prompt_length", 512)
    max_completion_length = config.get("max_completion_length", 128)
    batch_size = config.get("batch_size", 4)  # Prompts per batch
    grpo_epochs = config.get("grpo_epochs", 1)
    clip_eps = config.get("clip_epsilon", 0.2)
    beta = config.get("kl_beta", 0.04)  # KL penalty coefficient

    emit("status", phase="initializing", message="Setting up GRPO environment")

    if check_stop(stop_file):
        emit("status", phase="stopped", message="GRPO cancelled before start")
        return

    # --- Load model ---
    emit("status", phase="loading_model", message=f"Loading base model: {base_model}")

    try:
        from mlx_lm import load, generate as mlx_generate
    except ImportError as e:
        emit("error", message=f"Failed to import MLX: {e}")
        return

    if check_stop(stop_file):
        emit("status", phase="stopped", message="GRPO cancelled")
        return

    try:
        model, tokenizer = load(base_model, tokenizer_config={"trust_remote_code": True})
        total_params = sum(p.size for p in model.parameters() if hasattr(p, 'size'))
        emit("status", phase="model_loaded",
             message=f"Model loaded: {total_params/1e9 if total_params > 0 else 0:.1f}B parameters")
    except Exception as e:
        emit("error", message=f"Failed to load model: {e}")
        return

    # --- Load LoRA adapter if exists ---
    adapter_file = os.path.join(adapter_path, "adapters.safetensors")
    if os.path.exists(adapter_file):
        try:
            from mlx_lm.tuner.utils import load_adapters
            load_adapters(model, adapter_path)
            emit("status", phase="adapter_loaded",
                 message=f"Loaded LoRA adapter from {adapter_path}")
        except Exception as e:
            emit("status", phase="adapter_warning",
                 message=f"Could not load adapter (training from base): {e}")

    # --- Load training data ---
    emit("status", phase="loading_data", message=f"Loading data: {data_path}")
    if not os.path.exists(data_path):
        emit("error", message=f"Training data not found: {data_path}")
        return

    with open(data_path) as f:
        train_data = [json.loads(line) for line in f if line.strip()]

    if len(train_data) == 0:
        emit("error", message="Training data is empty")
        return

    emit("status", phase="data_loaded",
         message=f"Loaded {len(train_data)} prompts")

    # --- Setup optimizer ---
    # Freeze everything except LoRA layers
    model.freeze()
    # Unfreeze LoRA layers only
    for name, m in model.named_modules():
        if 'lora' in name.lower():
            m.unfreeze()

    trainable_params = sum(p.size for p in model.parameters() if hasattr(p, 'size') and not p.isfrozen())
    emit("status", phase="optimizer_setup",
         message=f"Trainable parameters: {trainable_params:,}")

    optimizer = optim.AdamW(learning_rate=lr)

    # State for compiled step
    state = [model.state, optimizer.state, mx.random.state]

    # --- GRPO Loss ---
    def grpo_loss_fn(model, input_ids, attention_mask, advantages):
        """GRPO loss with clipped surrogate objective and KL penalty."""
        logits = model(input_ids, attention_mask=attention_mask)
        # Simple log-prob of chosen tokens given previous context
        log_probs = mx.log(mx.softmax(logits, axis=-1))

        # Gather log probs of actual tokens
        batch_indices = mx.arange(input_ids.shape[1])[None, :]
        selected_log_probs = log_probs[batch_indices, input_ids]

        # Policy gradient loss with clipping
        ratio = mx.exp(selected_log_probs)  # Approximate probability ratio
        clipped_ratio = mx.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps)

        # Surrogate loss
        surrogate = -mx.minimum(
            ratio * advantages[:, None],
            clipped_ratio * advantages[:, None],
        )

        # KL penalty (approximate)
        # We use a simple KL estimate against a uniform reference
        policy_probs = mx.softmax(logits, axis=-1)
        uniform = mx.ones_like(policy_probs) / policy_probs.shape[-1]
        kl_div = mx.sum(policy_probs * (mx.log(policy_probs + 1e-8) - mx.log(uniform + 1e-8)), axis=-1)

        loss = mx.mean(surrogate) + beta * mx.mean(kl_div)
        return loss

    # --- Training loop ---
    emit("status", phase="grpo_started",
         message=f"Starting GRPO: {len(train_data)} prompts, group_size={group_size}")

    start_time = time.time()
    total_steps = min(len(train_data) // batch_size * grpo_epochs, 100)  # Cap steps
    step = 0

    for epoch in range(grpo_epochs):
        for batch_start in range(0, len(train_data), batch_size):
            if check_stop(stop_file):
                emit("status", phase="stopped", message="GRPO stopped by user")
                # Save checkpoint
                model.save_weights(os.path.join(adapter_path, "grpo_adapters.safetensors"))
                emit("checkpoint_saved", path=os.path.join(adapter_path, "grpo_adapters.safetensors"))
                return

            step += 1
            batch = train_data[batch_start:batch_start + batch_size]
            batch_losses = []

            for item in batch:
                prompt = item.get("prompt", item.get("question", ""))
                expected = item.get("completion", item.get("reference_answer", ""))

                # --- Generate group completions ---
                completions = []
                for g in range(group_size):
                    response = mlx_generate(
                        model, tokenizer,
                        prompt=prompt,
                        max_tokens=max_completion_length,
                        verbose=False,
                    )
                    completions.append(response)

                # --- Get rewards from LLM judge ---
                rewards = []
                for comp in completions:
                    reward = _get_judge_score(judge_model, prompt, expected, comp)
                    rewards.append(reward)

                # --- Compute advantages (group-normalized) ---
                rewards_t = mx.array(rewards, dtype=mx.float32)
                mean_r = mx.mean(rewards_t)
                std_r = mx.std(rewards_t) + 1e-8
                advantages = (rewards_t - mean_r) / std_r

                # --- Tokenize prompt + completion ---
                full_texts = [f"{prompt}\n{comp}" for comp in completions]
                encoded = tokenizer(full_texts, padding=True, truncation=True,
                                    max_length=max_prompt_length + max_completion_length,
                                    return_tensors="np")

                input_ids = mx.array(encoded["input_ids"])
                attention_mask = mx.array(encoded["attention_mask"])

                # --- Compute loss and update ---
                loss_and_grad = nn.value_and_grad(model, grpo_loss_fn)
                loss_val, grads = loss_and_grad(
                    model, input_ids, attention_mask, advantages
                )

                optimizer.update(model, grads)
                mx.eval(state)

                batch_losses.append(float(loss_val))

            # Emit progress
            avg_loss = sum(batch_losses) / len(batch_losses) if batch_losses else 0
            elapsed = time.time() - start_time
            progress = step / total_steps if total_steps > 0 else 0
            eta = elapsed / progress - elapsed if progress > 0 else 0

            emit("progress",
                 step=step,
                 total_steps=total_steps,
                 loss=round(avg_loss, 6),
                 lr=lr,
                 progress_percent=round(progress * 100, 1),
                 epoch=epoch + 1,
                 elapsed_seconds=round(elapsed),
                 eta_seconds=round(eta),
                 )

            if step >= total_steps:
                break
        else:
            continue
        break

    # --- Save ---
    emit("status", phase="saving", message="Saving GRPO adapter...")
    save_path = os.path.join(adapter_path, "grpo_adapters.safetensors")
    model.save_weights(save_path)

    metadata = {
        "niche": niche,
        "base_model": base_model,
        "judge_model": judge_model,
        "group_size": group_size,
        "learning_rate": lr,
        "total_steps": step,
        "final_loss": round(sum(batch_losses) / len(batch_losses), 6) if batch_losses else 0,
        "duration_seconds": round(time.time() - start_time),
        "type": "grpo",
    }
    with open(os.path.join(adapter_path, "grpo_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    emit("complete",
         output_dir=adapter_path,
         total_steps=step,
         duration_seconds=round(time.time() - start_time),
         )


def _get_judge_score(judge_model: str, prompt: str, expected: str, actual: str) -> int:
    """
    Query the LLM judge for binary reward.
    Returns 1 if the model's completion is correct, 0 otherwise.
    """
    import subprocess

    judge_prompt = f"""You are a strict judge. Evaluate whether the model's answer is correct.

QUESTION: {prompt}
EXPECTED ANSWER: {expected}
MODEL'S ANSWER: {actual}

Respond with exactly one number: 1 if the answer is correct, 0 if incorrect."""

    try:
        result = subprocess.run(
            ["cmd", "-t", "-m", judge_model, "-p", judge_prompt],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            output = result.stdout.strip()
            # Extract 1 or 0
            if "1" in output[:10]:
                return 1
            elif "0" in output[:10]:
                return 0
        return 0
    except Exception:
        return 0


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
