"""
Export adapter to GGUF and register with Ollama.
"""

import json
import os
import shutil
import subprocess
import sys
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

    # Resolve the base model the adapter was actually trained on: the adapter's own
    # adapter_config.json is authoritative (the global config.base_model may have
    # changed since training). Fall back to the configured base model.
    adapter_base = base_model
    cfg_path = os.path.join(adapter_path, "adapter_config.json")
    if os.path.exists(cfg_path):
        try:
            adapter_base = json.load(open(cfg_path)).get("model", base_model)
        except Exception:
            pass

    print(f"\nExporting {niche}...")
    print(f"  Adapter: {adapter_path}")
    print(f"  Base:    {adapter_base}")
    print(f"  Output:  {merged_dir}")

    # Fuse the LoRA adapter into the base model with mlx_lm's fuser. Unlike a bare
    # save_weights of the LoRA-applied model, this writes a *standalone* model dir
    # (config.json + fused weights + tokenizer + chat template) that the inference
    # server can load directly with mlx_lm.load — which save_weights alone cannot.
    os.makedirs(merged_dir, exist_ok=True)
    try:
        proc = subprocess.run(
            [
                sys.executable, "-m", "mlx_lm", "fuse",
                "--model", adapter_base,
                "--adapter-path", adapter_path,
                "--save-path", merged_dir,
            ],
            capture_output=True, text=True,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )
        if proc.returncode != 0 or not os.path.exists(os.path.join(merged_dir, "config.json")):
            print(f"  Export error: fuse failed (code {proc.returncode})")
            print((proc.stderr or "")[-500:])
            return None
        print(f"  Fused merged model: {merged_dir}")
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
