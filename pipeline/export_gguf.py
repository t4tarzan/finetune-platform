"""
Export adapter to GGUF and register with Ollama.
"""

import json
import os
import shutil
import subprocess
import yaml
from pathlib import Path


def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


def export_model(
    niche: str,
    adapter_path: str,
    export_dir: str = None,
    config: dict = None,
    register: bool = True,
):
    """Export LoRA adapter to merged model and optionally register with Ollama."""
    config = config or load_config()
    export_dir = export_dir or config.get("paths", {}).get("export_path", "models/gguf")
    os.makedirs(export_dir, exist_ok=True)

    base_model = config.get("base_model", "mlx-community/Qwen2.5-7B-Instruct-4bit")
    model_name = f"{niche}"
    merged_dir = os.path.join(export_dir, f"{model_name}_merged")

    print(f"\nExporting {niche}...")
    print(f"  Adapter: {adapter_path}")
    print(f"  Output: {merged_dir}")

    # Load model with adapter
    from mlx_lm import load
    from mlx_lm.tuner.utils import load_adapters

    try:
        model, tokenizer = load(base_model, tokenizer_config={"trust_remote_code": True})

        adapter_file = os.path.join(adapter_path, "adapters.safetensors")
        if os.path.exists(adapter_file):
            load_adapters(model, adapter_path)
            print(f"  Loaded adapter")
        else:
            print(f"  No adapter found, exporting base model only")

        # Save merged
        model.save_weights(os.path.join(merged_dir, "model.safetensors"))
        tokenizer.save_pretrained(merged_dir)
        print(f"  Saved merged model: {merged_dir}")

    except Exception as e:
        print(f"  Export error: {e}")
        return None

    if register:
        register_with_ollama(model_name, merged_dir)

    return merged_dir


def register_with_ollama(model_name: str, model_path: str):
    """Register a model with Ollama via Modelfile."""
    ollama_model = model_name.replace("_", "-").lower()
    modelfile = f"FROM {model_path}\n"

    modelfile_path = os.path.join(os.path.dirname(model_path), "Modelfile")
    with open(modelfile_path, "w") as f:
        f.write(modelfile)

    print(f"Registering '{ollama_model}' with Ollama...")
    result = subprocess.run(
        ["ollama", "create", ollama_model, "-f", modelfile_path],
        capture_output=True, text=True,
    )

    if result.returncode == 0:
        print(f"  ✓ '{ollama_model}' registered")
        return True
    else:
        print(f"  ✗ Registration failed: {result.stderr[:200]}")
        return False
