"""
Export a PEFT LoRA adapter to a merged HF model and register it with Ollama.

Linux counterpart to export_gguf.py (which is MLX-only). Mirrors the
export_model(niche, adapter_path, ...) signature so the platform can dispatch
to it when MLX is unavailable.
"""

import os

import yaml


def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


def _resolve_base_model(adapter_path: str, config: dict) -> str:
    """Prefer the base model recorded by the HF worker; fall back to a CPU default."""
    marker = os.path.join(adapter_path, "base_model.txt")
    if os.path.exists(marker):
        with open(marker) as f:
            val = f.read().strip()
            if val:
                return val
    # Match the worker's CPU default rather than config's MLX 4-bit id.
    return "Qwen/Qwen2.5-0.5B-Instruct"


def export_model(niche: str, adapter_path: str, export_dir: str = None,
                 config: dict = None, register: bool = True):
    """Merge LoRA adapter into the base model, save HF safetensors, register with Ollama."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    config = config or load_config()
    export_dir = export_dir or config.get("paths", {}).get("export_path", "models/gguf")
    os.makedirs(export_dir, exist_ok=True)

    base_model = _resolve_base_model(adapter_path, config)
    model_name = f"{niche}"
    merged_dir = os.path.join(export_dir, f"{model_name}_merged")
    os.makedirs(merged_dir, exist_ok=True)

    print(f"\nExporting {niche} (HF/CPU)...")
    print(f"  Base:    {base_model}")
    print(f"  Adapter: {adapter_path}")
    print(f"  Output:  {merged_dir}")

    # Fail fast if there's no adapter to merge — otherwise we'd silently export the
    # un-fine-tuned base model and report success (a confusing footgun, e.g. on a
    # mistyped adapter_path). Export here always follows training, so a missing
    # adapter is an error, not a request to export the base.
    adapter_cfg = os.path.join(adapter_path, "adapter_config.json")
    if not os.path.exists(adapter_cfg):
        print(f"  No adapter found at {adapter_path} (expected adapter_config.json)")
        return None

    model = base = tokenizer = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        base = AutoModelForCausalLM.from_pretrained(
            base_model, dtype=torch.float32, trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base, adapter_path)
        model = model.merge_and_unload()
        print("  Merged adapter into base weights")

        model.save_pretrained(merged_dir, safe_serialization=True)
        tokenizer.save_pretrained(merged_dir)
        print(f"  Saved merged model: {merged_dir}")
    except Exception as e:
        print(f"  Export error: {e}")
        return None
    finally:
        # Release the model from RAM promptly — on a memory-constrained box the
        # merged fp32 model is the bulk of the footprint, and registration below
        # loads its own copy in the inference-server process.
        import gc
        model = base = tokenizer = None
        gc.collect()

    if register:
        register_with_inference_server(model_name, os.path.abspath(merged_dir), config)

    return merged_dir


def register_with_inference_server(model_name: str, model_path: str, config: dict):
    """Load the merged model into the platform's inference server (port 7200).

    On Linux we serve via the in-repo inference server rather than Ollama, whose
    llama runner crashes on imported safetensors models.
    """
    import urllib.parse
    import urllib.request

    port = config.get("ports", {}).get("inference_api", 7200)
    served_name = model_name.replace("_", "-").lower()
    qs = urllib.parse.urlencode({"model_path": model_path, "model_name": served_name})
    url = f"http://127.0.0.1:{port}/api/manage/load?{qs}"
    print(f"Loading '{served_name}' into inference server on :{port}...")
    try:
        req = urllib.request.Request(url, data=b"", method="POST")
        resp = urllib.request.urlopen(req, timeout=180)
        body = resp.read().decode()
        print(f"  ✓ served on :{port} → {body}")
        return True
    except Exception as e:
        print(f"  ✗ inference-server load failed: {e}")
        print(f"    (model is merged at {model_path}; load it manually via /api/inference/load)")
        return False
