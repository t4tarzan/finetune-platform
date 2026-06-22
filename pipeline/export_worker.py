"""
Export Worker — runs a model export/merge in an isolated subprocess.

Merging a LoRA adapter loads a full model into RAM. Running it in the UI process
risks OOM-killing the whole server on a memory-constrained box. This worker
isolates that work in a fresh process (matching the platform's training/GRPO
subprocess design): an OOM here kills only this process, and the parent UI
server stays up and reports a clean error.

Protocol:
  stdin  : one JSON line {"niche": ..., "adapter_path": ...}
  stdout : exactly one JSON line — {"event": "complete", "merged_dir": ...}
                                 or {"event": "error", "message": ...}
           (all library/print noise is redirected to stderr so stdout stays clean)

Backend dispatch (MLX on Apple Silicon, HuggingFace elsewhere) mirrors the app.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    config_line = sys.stdin.readline().strip()
    real_stdout = sys.stdout
    # Keep stdout reserved for the single JSON result line; send everything the
    # export prints (and library output) to stderr.
    sys.stdout = sys.stderr
    try:
        config = json.loads(config_line) if config_line else {}
        niche = config.get("niche")
        adapter_path = config.get("adapter_path")
        if not niche or not adapter_path:
            result = {"event": "error", "message": "Missing niche or adapter_path"}
        else:
            from pipeline.training_manager import mlx_available
            if mlx_available():
                from pipeline.export_gguf import export_model
            else:
                from pipeline.export_hf import export_model
            merged = export_model(niche=niche, adapter_path=adapter_path, register=True)
            if merged:
                result = {"event": "complete", "merged_dir": merged}
            else:
                result = {"event": "error", "message": "Export returned no output (merge failed)."}
    except Exception as e:
        result = {"event": "error", "message": f"{type(e).__name__}: {e}"}
    finally:
        sys.stdout = real_stdout

    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
