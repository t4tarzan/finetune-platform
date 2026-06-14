"""
MLX LoRA Fine-Tuning Pipeline

Takes consensus-verified data, fine-tunes a base model using MLX LoRA,
then exports the adapter to GGUF and registers with Ollama.

Usage:
  python train_qlora.py --niche "my-domain" --epochs 3
"""

import json
import os
import sys
import shutil
import subprocess
import yaml
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.lora import TrainingArgs, train_model, load_dataset, linear_to_lora_layers
from mlx_lm.tuner.utils import load_adapters


def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


def prepare_training_data(
    verified_path: str,
    output_dir: str,
    test_split: float = 0.2,
    max_examples: int = None,
):
    """
    Convert consensus-verified jsonl data into MLX-LoRA training format.
    
    Input: verified_train.jsonl with {id, question, reference_answer, context, metadata}
    Output: train.jsonl, valid.jsonl in {prompt, completion} format
    """
    os.makedirs(output_dir, exist_ok=True)
    
    with open(verified_path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    
    if max_examples:
        rows = rows[:max_examples]
    
    formatted = []
    for row in rows:
        context = row.get("context", "")
        question = row["question"]
        answer = row["reference_answer"]
        
        # Wrap with context if available
        if context:
            prompt = f"Context: {context}\n\nQuestion: {question}"
        else:
            prompt = f"Question: {question}"
        
        formatted.append({
            "prompt": prompt,
            "completion": answer,
        })
    
    # Split into train/valid
    split_idx = int(len(formatted) * (1 - test_split))
    train = formatted[:split_idx]
    valid = formatted[split_idx:]
    
    train_path = os.path.join(output_dir, "train.jsonl")
    valid_path = os.path.join(output_dir, "valid.jsonl")
    
    with open(train_path, "w") as f:
        for row in train:
            f.write(json.dumps(row) + "\n")
    
    with open(valid_path, "w") as f:
        for row in valid:
            f.write(json.dumps(row) + "\n")
    
    print(f"Prepared {len(train)} training, {len(valid)} validation examples")
    print(f"  Train: {train_path}")
    print(f"  Valid: {valid_path}")
    
    return train_path, valid_path


def create_training_args(config: dict, niche: str):
    """Create a namespace-compatible args object for mlx_lm.lora."""
    training = config.get("training", {})
    
    class Args:
        pass
    
    args = Args()
    
    # Model
    args.model = config.get("base_model", "mlx-community/Qwen2.5-7B-Instruct-4bit")
    
    # Training flags
    args.train = True
    args.test = False
    args.fine_tune_type = "lora"
    args.optimizer = "adamw"
    args.optimizer_config = {"adamw": {}}
    args.mask_prompt = False
    args.num_layers = -1  # All layers
    args.batch_size = training.get("batch_size", 4)
    args.iters = training.get("iters", None)  # Will be computed from epochs
    args.val_batches = 25
    args.learning_rate = training.get("learning_rate", 1e-4)
    args.steps_per_report = 10
    args.steps_per_eval = 50
    args.save_every = 50
    args.max_seq_length = training.get("max_seq_length", 2048)
    args.grad_checkpoint = True
    args.grad_accumulation_steps = training.get("grad_accumulation_steps", 1)
    args.lr_schedule = None
    args.report_to = None
    args.project_name = niche
    args.seed = 42
    args.resume_adapter_file = None
    args.clear_cache_threshold = 0
    
    # LoRA params
    lora_rank = training.get("lora_rank", 16)
    lora_alpha = training.get("lora_alpha", 32)
    lora_scale = lora_alpha / lora_rank
    args.lora_parameters = {
        "rank": lora_rank,
        "dropout": 0.0,
        "scale": lora_scale,
    }
    
    return args


def compute_iters(dataset_size: int, epochs: int, batch_size: int) -> int:
    """Compute total iterations from dataset size, epochs and batch size."""
    steps_per_epoch = max(1, dataset_size // batch_size)
    return steps_per_epoch * epochs


def fine_tune(
    niche: str,
    data_dir: str,
    config: dict = None,
    epochs: int = None,
    adapter_path: str = None,
):
    """
    Run the full fine-tuning pipeline.
    
    Args:
        niche: Name for this model variant (e.g. "medical-coding")
        data_dir: Directory containing train.jsonl and valid.jsonl
        config: Pipeline config dict (loads from config.yaml if None)
        epochs: Override training epochs
        adapter_path: Where to save the LoRA adapter
    """
    config = config or load_config()
    training = config.get("training", {})
    epochs = epochs or training.get("epochs", 3)
    
    if adapter_path is None:
        adapter_path = os.path.join(
            config.get("paths", {}).get("adapter_path", "models/adapters"),
            niche,
        )
    
    # Create args
    args = create_training_args(config, niche)
    
    # Count training examples to compute iters
    train_path = os.path.join(data_dir, "train.jsonl")
    with open(train_path) as f:
        num_train = sum(1 for _ in f if _.strip())
    
    batch_size = args.batch_size
    args.iters = compute_iters(num_train, epochs, batch_size)
    args.adapter_path = adapter_path
    
    print(f"\n{'='*60}")
    print(f"Fine-Tuning: {niche}")
    print(f"{'='*60}")
    print(f"  Base model: {args.model}")
    print(f"  Training examples: {num_train}")
    print(f"  Epochs: {epochs}")
    print(f"  Iters: {args.iters}")
    print(f"  Batch size: {batch_size}")
    print(f"  LoRA rank: {training.get('lora_rank', 16)}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Adapter path: {adapter_path}")
    
    # Set data path for dataset loader
    args.data = data_dir
    
    # Load model
    print("\nLoading model...")
    model, tokenizer = load(args.model, tokenizer_config={"trust_remote_code": True})
    print(f"  Model loaded. Parameters: {sum(p.size for p in model.parameters() if hasattr(p, 'size')):,}")
    
    # Load datasets
    print("Loading datasets...")
    train_set, valid_set, _ = load_dataset(args, tokenizer)
    
    if len(train_set) == 0:
        print("ERROR: Training set is empty!")
        return False
    
    # Train
    print("\nStarting training...")
    train_model(args, model, train_set, valid_set)
    print("\nTraining complete!")
    
    # Save metadata
    metadata = {
        "niche": niche,
        "base_model": args.model,
        "epochs": epochs,
        "training_examples": num_train,
        "iters": args.iters,
        "lora_rank": training.get("lora_rank", 16),
        "learning_rate": args.learning_rate,
        "adapter_path": adapter_path,
    }
    
    meta_path = os.path.join(adapter_path, "training_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Training metadata saved to {meta_path}")
    
    return True


def export_to_gguf(
    niche: str,
    adapter_path: str,
    export_dir: str = None,
    config: dict = None,
):
    """
    Export merged model (base + LoRA) to GGUF format for Ollama.
    
    Uses mlx_lm.convert and then llama.cpp's convert-llm-ggml-to-gguf.py
    or saves directly as a merged MLX model that Ollama can reference.
    """
    config = config or load_config()
    export_dir = export_dir or config.get("paths", {}).get("export_path", "models/gguf")
    os.makedirs(export_dir, exist_ok=True)
    
    base_model = config.get("base_model", "mlx-community/Qwen2.5-7B-Instruct-4bit")
    model_name = f"{niche}-v1"
    gguf_path = os.path.join(export_dir, f"{model_name}.gguf")
    
    # Strategy: Merge the LoRA adapter into the MLX model and save as safetensors,
    # then use llama.cpp to convert to GGUF for Ollama
    merged_dir = os.path.join(export_dir, f"{model_name}_merged")
    
    print(f"\n{'='*60}")
    print(f"Exporting: {niche}")
    print(f"{'='*60}")
    
    # Load model with adapter
    print("Loading model with adapter for merge...")
    model, tokenizer = load(base_model, tokenizer_config={"trust_remote_code": True})
    
    # Apply LoRA adapters
    adapter_file = os.path.join(adapter_path, "adapters.safetensors")
    if os.path.exists(adapter_file):
        load_adapters(model, adapter_path)
        print(f"  Loaded adapter from {adapter_file}")
    else:
        print(f"  WARNING: Adapter not found at {adapter_file}")
        print(f"  Exporting base model only.")
    
    # Save merged model
    print(f"Saving merged model to {merged_dir}...")
    model.save_weights(os.path.join(merged_dir, "model.safetensors"))
    tokenizer.save_pretrained(merged_dir)
    print("  Merged model saved.")
    
    # Check for llama.cpp GGUF conversion tool
    gguf_convert_cmd = shutil.which("llama-quantize") or shutil.which("convert-llm-ggml-to-gguf.py")
    
    if gguf_convert_cmd:
        print(f"Converting to GGUF using {gguf_convert_cmd}...")
        # Note: This step requires llama.cpp to be installed
        try:
            subprocess.run([
                gguf_convert_cmd,
                "--input", merged_dir,
                "--output", gguf_path,
            ], check=True, capture_output=True, text=True)
            print(f"  GGUF exported: {gguf_path}")
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"  GGUF conversion skipped (install llama.cpp): {e}")
            print(f"  Merged MLX model available at: {merged_dir}")
    else:
        print("  llama.cpp not found — saving as MLX safetensors instead of GGUF")
        print(f"  Merged model: {merged_dir}")
        print("  Install llama.cpp for GGUF export: brew install llama.cpp")
    
    return merged_dir


def register_with_ollama(
    niche: str,
    model_path: str,
):
    """
    Register the fine-tuned model with Ollama via Modelfile.
    """
    model_name = f"{niche}-v1"
    
    # Check if using GGUF or MLX safetensors
    gguf_path = model_path if model_path.endswith(".gguf") else None
    merged_dir = model_path if not gguf_path else None
    
    if gguf_path:
        modelfile = f"FROM {gguf_path}\n"
    elif merged_dir:
        modelfile = f"FROM {merged_dir}\n"
    else:
        print(f"  Unknown model path format: {model_path}")
        return False
    
    modelfile_path = os.path.join(os.path.dirname(model_path), "Modelfile")
    with open(modelfile_path, "w") as f:
        f.write(modelfile)
    
    print(f"\nRegistering model '{model_name}' with Ollama...")
    result = subprocess.run(
        ["ollama", "create", model_name, "-f", modelfile_path],
        capture_output=True, text=True
    )
    
    if result.returncode == 0:
        print(f"  ✓ Model '{model_name}' registered with Ollama")
        
        # Verify it appears in the model list
        ls_result = subprocess.run(
            ["ollama", "list"],
            capture_output=True, text=True
        )
        if model_name in ls_result.stdout:
            print(f"  ✓ Verified: '{model_name}' available in Ollama")
        return True
    else:
        print(f"  ✗ Failed to register: {result.stderr}")
        return False


def run_full_pipeline(
    niche: str,
    verified_data_path: str,
    epochs: int = None,
    register: bool = True,
    config: dict = None,
):
    """
    Run the full pipeline: prepare data → fine-tune → export → register.
    
    Args:
        niche: Name for this model (e.g. "medical-coding")
        verified_data_path: Path to consensus-verified jsonl
        epochs: Number of training epochs
        register: Whether to register with Ollama
        config: Pipeline config
    """
    config = config or load_config()
    data_dir = config.get("paths", {}).get("data", "data")
    adapter_base = config.get("paths", {}).get("adapter_path", "models/adapters")
    export_base = config.get("paths", {}).get("export_path", "models/gguf")
    
    # Step 1: Prepare training data
    print("\n" + "="*60)
    print("STEP 1: Prepare Training Data")
    print("="*60)
    niche_data_dir = os.path.join(data_dir, niche)
    prepare_training_data(
        verified_path=verified_data_path,
        output_dir=niche_data_dir,
        test_split=config.get("eval", {}).get("test_split", 0.2),
    )
    
    # Step 2: Fine-tune
    print("\n" + "="*60)
    print("STEP 2: Fine-Tune")
    print("="*60)
    adapter_path = os.path.join(adapter_base, niche)
    success = fine_tune(
        niche=niche,
        data_dir=niche_data_dir,
        config=config,
        epochs=epochs,
        adapter_path=adapter_path,
    )
    
    if not success:
        print("Fine-tuning failed. Exiting.")
        return False
    
    # Step 3: Export
    print("\n" + "="*60)
    print("STEP 3: Export")
    print("="*60)
    model_path = export_to_gguf(
        niche=niche,
        adapter_path=adapter_path,
        export_dir=export_base,
        config=config,
    )
    
    # Step 4: Register with Ollama
    if register:
        print("\n" + "="*60)
        print("STEP 4: Register with Ollama")
        print("="*60)
        register_with_ollama(niche, model_path)
    
    print("\n" + "="*60)
    print(f"Pipeline complete for '{niche}'")
    print(f"  Adapter: {adapter_path}")
    print(f"  Exported: {model_path}")
    print("="*60)
    
    return True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="MLX LoRA Fine-Tuning Pipeline")
    parser.add_argument("--niche", type=str, required=True, help="Domain name (e.g. medical-coding)")
    parser.add_argument("--data", type=str, default="data/verified_train.jsonl", help="Path to verified training data")
    parser.add_argument("--epochs", type=int, default=None, help="Training epochs")
    parser.add_argument("--no-register", action="store_true", help="Skip Ollama registration")
    
    args = parser.parse_args()
    run_full_pipeline(
        niche=args.niche,
        verified_data_path=args.data,
        epochs=args.epochs,
        register=not args.no_register,
    )
