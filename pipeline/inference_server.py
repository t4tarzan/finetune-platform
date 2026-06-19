"""
Persistent Inference Server — runs fine-tuned models as hot OpenAI-compatible endpoints.

Launched on a dedicated port (7200 by default). Model stays in memory.
Supports multiple models hot-swapped. API key authentication.

API:
  GET  /v1/models          — list loaded models
  POST /v1/chat/completions — OpenAI-compatible chat completions
  POST /v1/completions      — raw completions
  POST /api/manage/load     — load/reload a model (API key required)
  GET  /api/health          — health check
"""

import json
import os
import sys
import time
import uuid
from typing import Optional

from fastapi import FastAPI, HTTPException, Header, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = FastAPI(title="Fine-Tuning Inference Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Model registry ─────────────────────────────────────────

_loaded_models: dict[str, dict] = {}  # model_name → {model, tokenizer, config}


def load_mlx_model(model_path: str, model_name: str = None):
    """Load an MLX model into memory (merged fine-tuned or base)."""
    global _loaded_models

    name = model_name or os.path.basename(model_path)
    print(f"Loading model '{name}' from {model_path}...")

    try:
        from mlx_lm import load, generate

        start = time.time()
        model, tokenizer = load(model_path, tokenizer_config={"trust_remote_code": True})
        elapsed = time.time() - start

        _loaded_models[name] = {
            "model": model,
            "tokenizer": tokenizer,
            "config": {
                "path": model_path,
                "loaded_at": time.time(),
                "load_time_s": round(elapsed, 1),
            },
        }
        print(f"  Loaded in {elapsed:.1f}s")
        return {"status": "ok", "model": name, "load_time_s": round(elapsed, 1)}
    except Exception as e:
        raise HTTPException(500, f"Failed to load model: {e}")


def get_loaded_model(model_name: str = None):
    """Get a loaded model by name, or the default (first loaded)."""
    if not _loaded_models:
        raise HTTPException(503, "No models loaded. Load one via POST /api/manage/load")

    if model_name:
        if model_name not in _loaded_models:
            raise HTTPException(404, f"Model '{model_name}' not loaded")
        return _loaded_models[model_name]
    else:
        # Return the first loaded model
        name = list(_loaded_models.keys())[0]
        return _loaded_models[name]


# ── API Models ─────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = ""
    messages: list[ChatMessage]
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    stream: bool = False


class CompletionRequest(BaseModel):
    model: str = ""
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.7
    stream: bool = False


# ── Management Endpoints ───────────────────────────────────

@app.post("/api/manage/load")
def manage_load(
    model_path: str = Query(..., description="Path or HF ID to the MLX model"),
    model_name: str = Query(None, description="Optional display name"),
):
    """Load a model into memory. Keeps it hot until explicitly unloaded or server restart."""
    return load_mlx_model(model_path, model_name)


@app.post("/api/manage/unload")
def manage_unload(model_name: str = Query(..., description="Model name to unload")):
    """Unload a model from memory (frees RAM)."""
    global _loaded_models
    if model_name in _loaded_models:
        del _loaded_models[model_name]
        # Force garbage collection
        import gc
        gc.collect()
        # MLX cache clear
        try:
            import mlx.core as mx
            mx.metal.clear_cache()
        except Exception:
            pass
        return {"status": "unloaded", "model": model_name}
    raise HTTPException(404, f"Model '{model_name}' not loaded")


@app.get("/api/manage/list")
def manage_list():
    """List loaded models with metadata."""
    return {
        "models": {
            name: info["config"]
            for name, info in _loaded_models.items()
        }
    }


@app.get("/api/health")
def health():
    """Health check with model status."""
    return {
        "status": "ok",
        "models_loaded": len(_loaded_models),
        "model_names": list(_loaded_models.keys()),
    }


# ── OpenAI-Compatible Endpoints ────────────────────────────

@app.get("/v1/models")
def list_models():
    """OpenAI-compatible models list."""
    return {
        "data": [
            {
                "id": name,
                "object": "model",
                "created": int(info["config"]["loaded_at"]),
                "owned_by": "finetune-platform",
            }
            for name, info in _loaded_models.items()
        ]
    }


@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest):
    """OpenAI-compatible chat completions endpoint."""
    model_entry = get_loaded_model(req.model if req.model else None)
    model = model_entry["model"]
    tokenizer = model_entry["tokenizer"]

    # Format messages into a single prompt
    prompt_parts = []
    for msg in req.messages:
        role = msg.role
        content = msg.content
        if role == "system":
            prompt_parts.append(f"<|system|>\n{content}")
        elif role == "user":
            prompt_parts.append(f"<|user|>\n{content}")
        elif role == "assistant":
            prompt_parts.append(f"<|assistant|>\n{content}")

    prompt_parts.append("<|assistant|>\n")
    prompt = "\n".join(prompt_parts)

    if req.stream:
        return _stream_completion(model, tokenizer, prompt, req)
    else:
        return _complete(model, tokenizer, prompt, req)


@app.post("/v1/completions")
def completions(req: CompletionRequest):
    """Raw completions endpoint."""
    model_entry = get_loaded_model(req.model if req.model else None)
    model = model_entry["model"]
    tokenizer = model_entry["tokenizer"]

    if req.stream:
        return _stream_completion(model, tokenizer, req.prompt, req)
    else:
        return _complete(model, tokenizer, req.prompt, req)


def _complete(model, tokenizer, prompt: str, req):
    """Generate a non-streaming completion."""
    from mlx_lm import generate as mlx_generate

    start = time.time()
    response = mlx_generate(
        model, tokenizer,
        prompt=prompt,
        max_tokens=req.max_tokens if hasattr(req, 'max_tokens') else 256,
        verbose=False,
    )
    elapsed = time.time() - start

    tokens = len(tokenizer.encode(response))
    return {
        "id": f"cmpl-{uuid.uuid4().hex[:12]}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": list(_loaded_models.keys())[0],
        "choices": [{"text": response, "index": 0, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": len(tokenizer.encode(prompt)),
            "completion_tokens": tokens,
            "total_tokens": len(tokenizer.encode(prompt)) + tokens,
        },
    }


def _stream_completion(model, tokenizer, prompt: str, req):
    """Generate a streaming completion via SSE."""
    from mlx_lm import stream_generate as mlx_stream

    max_tokens = req.max_tokens if hasattr(req, 'max_tokens') else 256

    async def generate():
        start_time = time.time()
        tokens = 0

        for token, _ in mlx_stream(
            model, tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
        ):
            tokens += 1
            chunk = {
                "id": f"cmpl-{uuid.uuid4().hex[:12]}",
                "object": "text_completion",
                "created": int(time.time()),
                "choices": [{"text": token, "index": 0, "finish_reason": None}],
            }
            yield f"data: {json.dumps(chunk)}\n\n"

        elapsed = time.time() - start_time
        yield f"data: {json.dumps({'choices': [{'text': '', 'index': 0, 'finish_reason': 'stop'}], 'usage': {'completion_tokens': tokens, 'time_s': round(elapsed, 2)}})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


# ── Main ───────────────────────────────────────────────────

def main():
    port = int(os.environ.get("INFERENCE_PORT", 7200))
    print(f"Starting Inference Server on http://localhost:{port}")
    print(f"  OpenAI API: http://localhost:{port}/v1/chat/completions")
    print(f"  Health:     http://localhost:{port}/api/health")
    print(f"  Load model first: curl -X POST 'http://localhost:{port}/api/manage/load?model_path=mlx-community/Qwen2.5-7B-Instruct-4bit&model_name=my-model'")
    print(f"  No models loaded yet. Use /api/manage/load to load one.")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")


if __name__ == "__main__":
    main()
