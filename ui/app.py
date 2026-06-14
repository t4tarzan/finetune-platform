"""
Fine-Tuning Platform — FastAPI server with Chat + Train UI.

Provides:
  /api/models      — list available models (Ollama)
  /api/chat        — chat with a model
  /api/leaderboard — benchmark data
  /api/train/*     — training orchestration (start, stop, status, progress, history)
  / — frontend with Chat + Train tabs
"""

import json
import os
import subprocess
import sys
import time
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.eval_harness import BenchmarkLeaderboard
from pipeline.training_manager import TrainingManager, load_config as load_train_config

app = FastAPI(title="Fine-Tuning Platform")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

leaderboard = BenchmarkLeaderboard()
train_manager = TrainingManager()
config = load_train_config()


# ── Request models ──────────────────────────────────────────

class StartTrainingRequest(BaseModel):
    niche: str
    dataset_type: str = "local"  # "local" or "bigset"
    dataset_desc: str = ""
    verified_data_path: str = ""
    base_model: str = ""
    lora_rank: int = 16
    lora_alpha: int = 32
    learning_rate: float = 1e-4
    batch_size: int = 4
    epochs: int = 3
    max_seq_length: int = 2048
    max_rows: int = 50


# ── Model endpoints ────────────────────────────────────────

@app.get("/api/models")
def list_models():
    models = []
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")[1:]
            for line in lines:
                parts = line.split()
                if parts:
                    name = parts[0]
                    is_ft = any(x in name for x in ("-iter", "-v1", "finetune"))
                    models.append({
                        "id": name, "name": name,
                        "provider": "ollama",
                        "type": "fine-tuned" if is_ft else "base",
                    })
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return {"models": models}


@app.get("/api/chat")
def chat(
    message: str = Query(...),
    model: str = Query(""),
    stream: bool = Query(False),
):
    if not model:
        raise HTTPException(400, "model required")
    models = list_models().get("models", [])
    is_ollama = any(m["id"] == model and m["provider"] == "ollama" for m in models)

    if is_ollama:
        return _chat_ollama(message, model, stream)
    else:
        return _chat_cmd(message, model)


def _chat_ollama(message: str, model: str, stream: bool):
    if stream:
        from fastapi.responses import StreamingResponse
        async def gen():
            proc = subprocess.Popen(
                ["ollama", "run", model],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True,
            )
            proc.stdin.write(message + "\n")
            proc.stdin.close()
            for line in proc.stdout:
                yield f"data: {json.dumps({'content': line})}\n\n"
            proc.wait()
            yield "data: [DONE]\n\n"
        return StreamingResponse(gen(), media_type="text/event-stream")

    start = time.time()
    result = subprocess.run(
        ["ollama", "run", model, message],
        capture_output=True, text=True, timeout=120
    )
    if result.returncode != 0:
        raise HTTPException(500, result.stderr[:200])
    return {
        "response": result.stdout.strip(), "model": model,
        "latency_ms": round((time.time() - start) * 1000),
    }


def _chat_cmd(message: str, model: str):
    start = time.time()
    result = subprocess.run(
        ["cmd", "-t", "-m", model, "-p", message],
        capture_output=True, text=True, timeout=60
    )
    if result.returncode != 0:
        raise HTTPException(500, result.stderr[:200])
    return {
        "response": result.stdout.strip(), "model": model,
        "latency_ms": round((time.time() - start) * 1000),
        "provider": "cmd",
    }


# ── Leaderboard ────────────────────────────────────────────

@app.get("/api/leaderboard")
def get_leaderboard(niche: Optional[str] = Query(None)):
    data = leaderboard.load()
    if niche:
        return data.get(niche, {})
    return data


# ── Training endpoints ─────────────────────────────────────

@app.post("/api/train/start")
async def train_start(req: StartTrainingRequest):
    if train_manager.is_training:
        raise HTTPException(400, "Training already in progress")

    result = await train_manager.start_training(req.model_dump())
    if "error" in result:
        raise HTTPException(500, result["error"])
    return result


@app.get("/api/train/status")
def train_status():
    status = train_manager.get_status()
    return status


@app.post("/api/train/stop")
async def train_stop(save: bool = Query(False, description="Save checkpoint before stopping")):
    result = await train_manager.stop_training(save_checkpoint=save)
    # Trigger export if we saved
    if save and train_manager.current_run:
        from pipeline.export_gguf import export_model
        export_model(
            niche=train_manager.current_run.get("niche", "model"),
            adapter_path=train_manager.current_run.get("output_dir", "models/adapters/default"),
        )
    return result


@app.get("/api/train/progress")
async def train_progress():
    """SSE stream of training progress events."""
    async def event_generator():
        async for event in train_manager.event_stream():
            yield f"data: {json.dumps(event)}\n\n"
        yield "data: {\"event\": \"stream_end\"}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/train/history")
def train_history(limit: int = Query(20, ge=1, le=100)):
    return {"runs": train_manager.get_history(limit=limit)}


@app.get("/api/train/niches")
def list_niches():
    """List available fine-tuned niches / datasets from the data directory."""
    data_dir = config.get("paths", {}).get("data", "data")
    niches = []
    if os.path.exists(data_dir):
        for entry in os.listdir(data_dir):
            full = os.path.join(data_dir, entry)
            if os.path.isdir(full):
                # Check for training data
                has_train = os.path.exists(os.path.join(full, "verified_train.jsonl"))
                niches.append({
                    "name": entry,
                    "has_training_data": has_train,
                    "path": full,
                })
    return {"niches": niches}


# ── Export ──────────────────────────────────────────────────

@app.post("/api/export")
def export_model_endpoint(
    niche: str = Query(...),
    adapter_path: str = Query(...),
):
    from pipeline.export_gguf import export_model
    export_model(niche=niche, adapter_path=adapter_path, register=True)
    return {"status": "ok", "niche": niche}


# ── Documentation ──────────────────────────────────────────

V1_DOCS = {
    "version": "1.0.0",
    "name": "Fine-Tuning Platform",
    "description": "Local recursive fine-tuning platform on M5 Ultra (128GB). "
                   "Generates datasets, verifies via multi-model consensus, "
                   "fine-tunes with MLX LoRA, benchmarks, and serves models via Ollama.",
    "architecture": {
        "backend": "FastAPI (Python 3.13) on port 7100",
        "ml_framework": "MLX (Apple Silicon native, uses unified memory)",
        "training_isolation": "Subprocess via subprocess.Popen with JSONL stdout IPC",
        "progress_streaming": "Server-Sent Events (SSE) with event types: status/progress/complete/error",
        "data_format": "JSONL with {question, reference_answer, context}",
        "model_format": "MLX safetensors → GGUF → Ollama registration",
        "port_range": "7000-7500 for all built services",
    },
    "features": {
        "chat": {
            "endpoint": "GET /api/chat",
            "description": "Chat with any Ollama-registered or commandcode API model",
            "model_dropdown": "Auto-populated from Ollama, groups fine-tuned vs base models",
            "provider_support": "Ollama (local) and cmd (API)",
        },
        "training": {
            "endpoints": {
                "start": "POST /api/train/start",
                "stop": "POST /api/train/stop",
                "status": "GET /api/train/status",
                "progress": "GET /api/train/progress (SSE)",
                "history": "GET /api/train/history",
                "niches": "GET /api/train/niches",
            },
            "worker": "pipeline/training_worker.py — isolated subprocess, emits progress via stdout JSONL",
            "configurable_params": [
                "niche (domain name)",
                "dataset_type (local/bigset)",
                "base_model (HuggingFace MLX path)",
                "lora_rank (1-256)",
                "lora_alpha (1-512)",
                "batch_size (1-128)",
                "learning_rate",
                "epochs (1-100)",
                "max_seq_length",
                "max_rows",
            ],
            "live_metrics": ["loss", "learning_rate", "epoch", "progress_percent", "eta_seconds", "tokens_per_sec"],
            "stop_behavior": "Touch stop file → callback checks every 3 steps → saves partial checkpoint",
        },
        "consensus_verification": {
            "description": "Validates training data across 3+ diverse commandcode API models",
            "models_used": ["deepseek/deepseek-v4-pro", "Qwen/Qwen3.6-Max-Preview", "moonshotai/Kimi-K2.7-Code", "claude-sonnet-4-6"],
            "threshold": "≥3 models must agree, average confidence ≥ 0.7",
            "output": "verified_train.jsonl + rejected_train.jsonl + consensus_report.json",
            "module": "pipeline/consensus_verifier.py",
        },
        "benchmark_leaderboard": {
            "endpoint": "GET /api/leaderboard",
            "metrics": ["accuracy", "grounding", "consistency"],
            "persistence": "benchmarks/leaderboard.json",
            "tracking": "Baseline → iteration deltas with improvement threshold stopping",
            "module": "pipeline/eval_harness.py",
        },
        "export": {
            "endpoint": "POST /api/export",
            "format": "MLX safetensors → Ollama GGUF via Modelfile",
            "auto_register": "Registered with Ollama after export, appears in model dropdown",
            "module": "pipeline/export_gguf.py",
        },
        "recursive_loop": {
            "description": "Ties together data generation, consensus, training, eval in automated iterations",
            "termination": "Stops when accuracy ≥95%, improvement <1% for 2 iterations, or max iterations reached",
            "module": "pipeline/recursive_loop.py",
            "usage": "python pipeline/recursive_loop.py --niche-name <name> --niche-desc '<description>' --max-iterations 5",
        },
    },
    "pipeline_modules": {
        "config.yaml": "Central configuration (models, thresholds, hyperparams, paths, ports)",
        "consensus_verifier.py": "Multi-model data verification with DataPoint, ModelVerdict, ConsensusResult types",
        "train_qlora.py": "Data preparation + MLX LoRA fine-tuning wrapper",
        "training_worker.py": "Subprocess entry point — handles model loading, LoRA, training loop, progress emission",
        "training_manager.py": "Manager — spawns worker, reads stdout JSONL, manages state, SSE streaming, history persistence",
        "eval_harness.py": "Benchmark runner against held-out test sets with EvalResult and BenchmarkScore types",
        "export_gguf.py": "Merges LoRA adapter, saves as MLX safetensors, registers with Ollama",
        "recursive_loop.py": "Recursive orchestrator with BigSet integration, automatic iteration, and convergence detection",
    },
    "ui": {
        "tabs": ["Chat", "Train"],
        "chat_tab": "Model selector dropdown → message input → streaming response via Ollama API",
        "train_tab": "Config form (left) + live training view (right) with loss chart, progress bar, metric grid, history",
        "sidebar": "Model selector + benchmark leaderboard with per-niche accuracy tracking and iteration deltas",
        "port": 7100,
    },
    "hardware": {
        "platform": "Apple M5 Max",
        "ram": "128GB unified memory",
        "ml_framework": "MLX (metal available: True)",
        "notes": "128GB fits 70B-class 4-bit models (~45GB) + TurboVec index (~4GB) + training cache with room to spare",
    },
    "data_flow": {
        "step_1": "BigSet or local JSONL → raw dataset",
        "step_2": "4 commandcode API models verify each row → only consensus-passing rows kept",
        "step_3": "Verified data split into train/valid JSONL in {prompt, completion} format",
        "step_4": "MLX-LoRA subprocess fine-tunes base model, streams loss/progress via SSE",
        "step_5": "Adapter merged with base → exported as GGUF → registered with Ollama",
        "step_6": "Model appears in dropdown → chat or benchmark → gaps drive next iteration",
    },
    "external_dependencies": {
        "ollama": "brew install ollama (model serving)",
        "bigset": "npm install --global @adamexu/bigset (dataset generation)",
        "mlx": "Apple-native ML framework (pip install mlx mlx-lm)",
        "turbovec": "Vector compression for RAG (pip install turbovec)",
        "commandcode_api": "cmd --list-models (30+ models available for consensus)",
    },
    "taste_configuration": {
        "ports": "All built services use 7000-7500 range",
        "learning": "Taste learns from each session and improves code generation",
        "consensus": "Multi-model verification reduces hallucination in training data",
    },
}


@app.get("/api/docs/v1")
def get_docs_v1():
    """Return the full v1 platform documentation."""
    return V1_DOCS


# ── Frontend ───────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(HTML_TEMPLATE)


# ── HTML Template ──────────────────────────────────────────

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Fine-Tuning Platform — localhost:7100</title>
<style>
  :root {
    --bg: #0d1117; --surface: #161b22; --surface2: #1c2128;
    --border: #30363d; --text: #e6edf3; --text-secondary: #8b949e;
    --accent: #58a6ff; --accent-hover: #79c0ff;
    --success: #3fb950; --warning: #d29922; --danger: #f85149;
    --radius: 8px;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: var(--bg); color: var(--text); height: 100vh; display: flex;
  }
  .sidebar {
    width: 280px; background: var(--surface);
    border-right: 1px solid var(--border);
    display: flex; flex-direction: column; overflow-y: auto;
  }
  .sidebar-header {
    padding: 16px; border-bottom: 1px solid var(--border);
    font-weight: 600; font-size: 14px; text-transform: uppercase;
    letter-spacing: 0.5px; color: var(--text-secondary);
  }
  .model-selector { padding: 12px 16px; border-bottom: 1px solid var(--border); }
  .model-selector label { font-size: 12px; color: var(--text-secondary); display: block; margin-bottom: 6px; }
  .model-selector select {
    width: 100%; padding: 8px; border-radius: var(--radius);
    border: 1px solid var(--border); background: var(--surface2);
    color: var(--text); font-size: 13px; cursor: pointer;
  }
  .model-selector select:focus { outline: none; border-color: var(--accent); }
  .leaderboard-section { padding: 12px 16px; flex: 1; }
  .leaderboard-section h3 { font-size: 12px; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 12px; }
  .niche-entry { background: var(--surface2); border-radius: var(--radius); padding: 12px; margin-bottom: 8px; }
  .niche-name { font-size: 13px; font-weight: 600; margin-bottom: 8px; }
  .metric { display: flex; justify-content: space-between; font-size: 12px; padding: 3px 0; }
  .metric-label { color: var(--text-secondary); }
  .metric-value { font-weight: 500; }
  .metric-value.positive { color: var(--success); }
  .metric-value.negative { color: var(--danger); }
  .delta { font-size: 11px; margin-left: 4px; }
  .status-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: var(--success); margin-right: 6px; }

  /* ── Docs section ── */
  .docs-section { border-top: 1px solid var(--border); padding: 0; }
  .docs-header {
    padding: 12px 16px; cursor: pointer; font-size: 12px;
    color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.5px;
    display: flex; justify-content: space-between; align-items: center;
    user-select: none;
  }
  .docs-header:hover { background: var(--surface2); color: var(--text); }
  .docs-header .arrow { transition: transform 0.2s; font-size: 10px; }
  .docs-header .arrow.open { transform: rotate(90deg); }
  .docs-body { padding: 0 16px 12px; font-size: 12px; color: var(--text-secondary); line-height: 1.5; display: none; }
  .docs-body.open { display: block; }
  .docs-body h4 { color: var(--text); font-weight: 600; margin: 8px 0 4px; font-size: 12px; }
  .docs-body ul { padding-left: 14px; margin: 4px 0; }
  .docs-body li { margin: 2px 0; }
  .docs-body code { background: var(--surface2); padding: 1px 4px; border-radius: 3px; font-size: 11px; }
  .docs-tag { display: inline-block; padding: 1px 5px; border-radius: 3px; font-size: 10px; margin-right: 4px; }
  .docs-tag.api { background: rgba(88,166,255,0.15); color: var(--accent); }
  .docs-tag.ml { background: rgba(63,185,80,0.15); color: var(--success); }

  .main-panel { flex: 1; display: flex; flex-direction: column; min-width: 0; }
  .tabs { display: flex; border-bottom: 1px solid var(--border); background: var(--surface); }
  .tab {
    padding: 12px 24px; cursor: pointer; font-size: 14px; font-weight: 500;
    color: var(--text-secondary); border-bottom: 2px solid transparent;
    transition: all 0.15s; user-select: none;
  }
  .tab:hover { color: var(--text); background: var(--surface2); }
  .tab.active { color: var(--accent); border-bottom-color: var(--accent); }

  .tab-content { display: none; flex: 1; flex-direction: column; }
  .tab-content.active { display: flex; }

  .chat-area { flex: 1; overflow-y: auto; padding: 24px; display: flex; flex-direction: column; gap: 16px; }
  .message { max-width: 80%; padding: 12px 16px; border-radius: var(--radius); line-height: 1.5; font-size: 14px; white-space: pre-wrap; word-wrap: break-word; }
  .message.user { background: var(--accent); color: #fff; align-self: flex-end; border-bottom-right-radius: 4px; }
  .message.assistant { background: var(--surface2); border: 1px solid var(--border); align-self: flex-start; border-bottom-left-radius: 4px; }
  .message .meta { font-size: 11px; color: var(--text-secondary); margin-top: 8px; padding-top: 8px; border-top: 1px solid var(--border); }
  .input-area { padding: 16px 24px; border-top: 1px solid var(--border); display: flex; gap: 8px; }
  .input-area input { flex: 1; padding: 10px 16px; border-radius: var(--radius); border: 1px solid var(--border); background: var(--surface2); color: var(--text); font-size: 14px; }
  .input-area input:focus { outline: none; border-color: var(--accent); }
  .input-area button { padding: 10px 20px; border-radius: var(--radius); border: none; background: var(--accent); color: #fff; font-size: 14px; font-weight: 500; cursor: pointer; }
  .input-area button:hover { background: var(--accent-hover); }
  .input-area button:disabled { opacity: 0.5; cursor: not-allowed; }
  .empty-state { display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100%; color: var(--text-secondary); gap: 8px; }
  .loading { display: inline-block; width: 16px; height: 16px; border: 2px solid var(--border); border-top-color: var(--accent); border-radius: 50%; animation: spin 0.6s linear infinite; }
  @keyframes spin { to { transform: rotate(360deg); } }

  /* ── Train tab ── */
  .train-layout { display: flex; flex: 1; overflow: hidden; }
  .train-config { width: 380px; padding: 20px; border-right: 1px solid var(--border); overflow-y: auto; }
  .train-config h3 { font-size: 13px; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 16px; }
  .form-group { margin-bottom: 14px; }
  .form-group label { display: block; font-size: 12px; color: var(--text-secondary); margin-bottom: 4px; }
  .form-group input, .form-group select {
    width: 100%; padding: 8px 10px; border-radius: var(--radius);
    border: 1px solid var(--border); background: var(--surface2);
    color: var(--text); font-size: 13px;
  }
  .form-group input:focus, .form-group select:focus { outline: none; border-color: var(--accent); }
  .form-row { display: flex; gap: 10px; }
  .form-row .form-group { flex: 1; }
  .form-section { margin: 16px 0 12px; padding-top: 12px; border-top: 1px solid var(--border); }
  .form-section h4 { font-size: 12px; color: var(--text-secondary); margin-bottom: 10px; }
  .btn { padding: 10px 20px; border-radius: var(--radius); border: none; font-size: 14px; font-weight: 500; cursor: pointer; width: 100%; }
  .btn-primary { background: var(--accent); color: #fff; }
  .btn-primary:hover { background: var(--accent-hover); }
  .btn-primary:disabled { opacity: 0.5; cursor: not-allowed; }
  .btn-danger { background: var(--danger); color: #fff; }
  .btn-danger:hover { opacity: 0.9; }
  .btn-success { background: var(--success); color: #fff; }
  .btn-outline { background: transparent; border: 1px solid var(--border); color: var(--text); }
  .btn-outline:hover { border-color: var(--accent); color: var(--accent); }

  .train-view { flex: 1; display: flex; flex-direction: column; overflow: hidden; }
  .train-status-bar { padding: 16px 20px; border-bottom: 1px solid var(--border); display: flex; align-items: center; gap: 12px; }
  .train-status-bar .phase { font-size: 13px; font-weight: 500; }
  .train-status-bar .message { font-size: 12px; color: var(--text-secondary); background: none; border: none; padding: 0; max-width: 100%; }
  .train-charts { flex: 1; overflow-y: auto; padding: 20px; }
  .chart-container { background: var(--surface2); border-radius: var(--radius); padding: 16px; margin-bottom: 16px; }
  .chart-container h4 { font-size: 12px; color: var(--text-secondary); margin-bottom: 8px; }
  .chart-canvas { width: 100%; height: 160px; position: relative; }
  .chart-canvas canvas { width: 100%; height: 100%; }
  .metric-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-bottom: 16px; }
  .metric-card { background: var(--surface2); border-radius: var(--radius); padding: 12px; text-align: center; }
  .metric-card .value { font-size: 20px; font-weight: 600; }
  .metric-card .label { font-size: 11px; color: var(--text-secondary); margin-top: 2px; }
  .progress-bar { height: 6px; background: var(--border); border-radius: 3px; overflow: hidden; margin: 8px 0; }
  .progress-bar .fill { height: 100%; background: var(--accent); border-radius: 3px; transition: width 0.3s; }

  .history-view { flex: 1; overflow-y: auto; padding: 20px; }
  .history-entry { background: var(--surface2); border-radius: var(--radius); padding: 12px 16px; margin-bottom: 8px; display: flex; justify-content: space-between; align-items: center; }
  .history-entry .h-left { flex: 1; }
  .history-entry .h-niche { font-size: 13px; font-weight: 600; }
  .history-entry .h-meta { font-size: 11px; color: var(--text-secondary); margin-top: 2px; }
  .history-entry .h-status { font-size: 12px; padding: 3px 8px; border-radius: 4px; }
  .h-status.completed { background: rgba(63,185,80,0.15); color: var(--success); }
  .h-status.error { background: rgba(248,81,73,0.15); color: var(--danger); }
  .h-status.stopped { background: rgba(210,153,34,0.15); color: var(--warning); }
</style>
</head>
<body>
<div class="sidebar">
  <div class="sidebar-header"><span class="status-dot"></span> Fine-Tuning Platform</div>
  <div class="model-selector">
    <label for="model-select">Active Model</label>
    <select id="model-select" onchange="switchModel()">
      <option value="">Loading models...</option>
    </select>
  </div>
  <div class="leaderboard-section">
    <h3>Leaderboard</h3>
    <div id="leaderboard-content">Loading...</div>
  </div>
  <div class="docs-section">
    <div class="docs-header" onclick="toggleDocs()">
      <span>v1.0 Documentation</span>
      <span class="arrow" id="docs-arrow">▶</span>
    </div>
    <div class="docs-body" id="docs-body"></div>
  </div>
</div>
<div class="main-panel">
  <div class="tabs">
    <div class="tab active" onclick="switchTab('chat',this)">Chat</div>
    <div class="tab" id="train-tab" onclick="switchTab('train',this)">Train</div>
  </div>

  <!-- Chat tab -->
  <div class="tab-content active" id="tab-chat">
    <div class="main-header" style="padding:12px 24px;border-bottom:1px solid var(--border);display:flex;align-items:center;gap:8px;">
      <span style="font-size:16px;font-weight:600;">Chat</span>
      <span style="font-size:12px;color:var(--text-secondary);font-weight:400;" id="current-model-label">no model selected</span>
    </div>
    <div class="chat-area" id="chat-area">
      <div class="empty-state">
        <div>Select a model and start chatting</div>
        <div class="key-hint" style="font-size:12px;padding:4px 8px;background:var(--surface2);border-radius:4px;border:1px solid var(--border);">Fine-tuned models appear in the dropdown automatically</div>
      </div>
    </div>
    <div class="input-area">
      <input type="text" id="message-input" placeholder="Type a message..." onkeydown="if(event.key==='Enter') sendMessage()"/>
      <button id="send-btn" onclick="sendMessage()">Send</button>
    </div>
  </div>

  <!-- Train tab -->
  <div class="tab-content" id="tab-train">
    <div class="train-layout">
      <!-- Config sidebar -->
      <div class="train-config">
        <h3>Training Configuration</h3>

        <div class="form-section"><h4>Dataset</h4></div>
        <div class="form-group">
          <label>Niche / Domain Name</label>
          <input id="ft-niche" value="my-domain" placeholder="e.g. medical-coding" />
        </div>
        <div class="form-group">
          <label>Dataset Type</label>
          <select id="ft-dataset-type">
            <option value="local">Local JSONL (verified_train.jsonl)</option>
            <option value="bigset">BigSet (generate from description)</option>
          </select>
        </div>
        <div class="form-group" id="ft-data-path-group">
          <label>Verified Data Path</label>
          <input id="ft-data-path" value="data/verified_train.jsonl" placeholder="path to verified_train.jsonl" />
        </div>
        <div class="form-group" id="ft-desc-group" style="display:none">
          <label>Dataset Description (for BigSet)</label>
          <input id="ft-desc" placeholder="AI startups in SF hiring engineers..." />
        </div>

        <div class="form-section"><h4>Model</h4></div>
        <div class="form-group">
          <label>Base Model</label>
          <select id="ft-base-model">
            <option value="mlx-community/Qwen2.5-7B-Instruct-4bit">Qwen2.5-7B (4-bit)</option>
            <option value="mlx-community/Mistral-7B-Instruct-v0.3-4bit">Mistral-7B (4-bit)</option>
            <option value="mlx-community/Llama-3.2-3B-Instruct-4bit">Llama 3.2-3B (4-bit)</option>
          </select>
        </div>

        <div class="form-section"><h4>Hyperparameters</h4></div>
        <div class="form-row">
          <div class="form-group"><label>LoRA Rank</label><input id="ft-lora-rank" value="16" type="number" min="1" max="256" /></div>
          <div class="form-group"><label>LoRA Alpha</label><input id="ft-lora-alpha" value="32" type="number" min="1" max="512" /></div>
        </div>
        <div class="form-row">
          <div class="form-group"><label>Batch Size</label><input id="ft-batch" value="4" type="number" min="1" max="128" /></div>
          <div class="form-group"><label>Learning Rate</label><input id="ft-lr" value="0.0001" step="0.00001" /></div>
        </div>
        <div class="form-row">
          <div class="form-group"><label>Epochs</label><input id="ft-epochs" value="3" type="number" min="1" max="100" /></div>
          <div class="form-group"><label>Max Rows</label><input id="ft-rows" value="50" type="number" min="1" max="10000" /></div>
        </div>

        <div style="margin-top:20px;display:flex;gap:8px;flex-direction:column;">
          <button class="btn btn-primary" id="ft-start-btn" onclick="startTraining()">Start Training</button>
          <button class="btn btn-danger" id="ft-stop-btn" onclick="stopTraining()" style="display:none">Stop Training</button>
          <button class="btn btn-outline" onclick="exportModel()">Export & Register with Ollama</button>
        </div>
      </div>

      <!-- View area -->
      <div class="train-view">
        <div class="train-status-bar">
          <span id="ft-phase" style="font-size:13px;font-weight:500;">Idle</span>
          <span id="ft-message" style="font-size:12px;color:var(--text-secondary);">Ready to train</span>
        </div>
        <div id="train-content" style="flex:1;overflow-y:auto;padding:20px;">
          <div class="empty-state" id="train-empty">
            <div>Configure your training job and click Start</div>
          </div>
          <!-- Metrics grid (hidden until training starts) -->
          <div id="train-metrics" style="display:none">
            <div class="metric-grid">
              <div class="metric-card"><div class="value" id="m-loss">—</div><div class="label">Loss</div></div>
              <div class="metric-card"><div class="value" id="m-lr">—</div><div class="label">Learning Rate</div></div>
              <div class="metric-card"><div class="value" id="m-epoch">—</div><div class="label">Epoch</div></div>
              <div class="metric-card"><div class="value" id="m-eta">—</div><div class="label">ETA</div></div>
            </div>
            <div class="progress-bar"><div class="fill" id="train-progress" style="width:0%"></div></div>
            <div class="chart-container">
              <h4>Training Loss</h4>
              <div class="chart-canvas"><canvas id="loss-chart"></canvas></div>
            </div>
            <div class="chart-container">
              <h4>History</h4>
              <div id="train-history-entries"></div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</div>

<script>
// ── Tab switching ──
let currentModel = '';
let trainingEventSource = null;
let lossData = [];

function switchTab(name, el) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
  el.classList.add('active');
  document.getElementById('tab-' + name).classList.add('active');
}

document.getElementById('ft-dataset-type').addEventListener('change', function() {
  const isBigset = this.value === 'bigset';
  document.getElementById('ft-data-path-group').style.display = isBigset ? 'none' : '';
  document.getElementById('ft-desc-group').style.display = isBigset ? '' : 'none';
});

// ── Models ──
async function loadModels() {
  try {
    const res = await fetch('/api/models');
    const data = await res.json();
    const select = document.getElementById('model-select');
    select.innerHTML = '';
    const ft = data.models.filter(m => m.type === 'fine-tuned');
    const base = data.models.filter(m => m.type === 'base');
    if (ft.length) { const og = document.createElement('optgroup'); og.label = 'Fine-Tuned';
      ft.forEach(m => { const o = document.createElement('option'); o.value = m.id; o.textContent = m.name; og.appendChild(o); }); select.appendChild(og); }
    if (base.length) { const og = document.createElement('optgroup'); og.label = 'Base Models';
      base.forEach(m => { const o = document.createElement('option'); o.value = m.id; o.textContent = m.name; og.appendChild(o); }); select.appendChild(og); }
    if (!data.models.length) select.innerHTML = '<option value="">No models available</option>';
  } catch(e) { console.error(e); }
}

function switchModel() {
  currentModel = document.getElementById('model-select').value;
  document.getElementById('current-model-label').textContent = currentModel || 'no model selected';
}

// ── Chat ──
async function sendMessage() {
  const input = document.getElementById('message-input');
  const msg = input.value.trim();
  if (!msg || !currentModel) return;
  input.value = '';
  document.getElementById('send-btn').disabled = true;
  const ca = document.getElementById('chat-area');
  const empty = ca.querySelector('.empty-state');
  if (empty) empty.remove();
  const ud = document.createElement('div'); ud.className = 'message user'; ud.textContent = msg; ca.appendChild(ud);
  const ld = document.createElement('div'); ld.className = 'message assistant'; ld.innerHTML = '<div class="loading"></div> Thinking...'; ca.appendChild(ld);
  ca.scrollTop = ca.scrollHeight;
  try {
    const r = await fetch('/api/chat?message='+encodeURIComponent(msg)+'&model='+encodeURIComponent(currentModel));
    const d = await r.json();
    ld.innerHTML = d.response;
    if (d.latency_ms) { const m = document.createElement('div'); m.className = 'meta'; m.textContent = currentModel+' ('+d.latency_ms+'ms)'; ld.appendChild(m); }
  } catch(e) { ld.innerHTML = 'Error: '+e.message; }
  ca.scrollTop = ca.scrollHeight;
  document.getElementById('send-btn').disabled = false;
}

// ── Leaderboard ──
async function loadLeaderboard() {
  try {
    const res = await fetch('/api/leaderboard'); const data = await res.json();
    const c = document.getElementById('leaderboard-content'); const niches = Object.keys(data);
    if (!niches.length) { c.innerHTML = '<div style="color:var(--text-secondary);font-size:12px">No benchmarks yet.</div>'; return; }
    c.innerHTML = niches.map(n => {
      const e = data[n]; const bl = e.baseline || {}; const its = e.iterations || []; const last = its[its.length-1];
      const r = last ? last.results : bl; const d = last ? last.delta : {};
      const fmt = v => (v*100).toFixed(1)+'%';
      const df = v => { if(v===undefined)return''; const s=v>=0?'+':''; const c=v>0.01?'positive':v<-0.01?'negative':'neutral'; return '<span class="delta '+c+'">'+s+(v*100).toFixed(1)+'%</span>'; };
      return '<div class="niche-entry"><div class="niche-name">'+n+'</div><div class="metric"><span class="metric-label">Accuracy</span><span class="metric-value">'+fmt(r.accuracy)+' '+df(d.accuracy)+'</span></div><div class="metric"><span class="metric-label">Iters</span><span class="metric-value">'+its.length+'</span></div></div>';
    }).join('');
  } catch(e) { console.error(e); }
}

// ── Training ──
function getConfig() {
  return {
    niche: document.getElementById('ft-niche').value,
    dataset_type: document.getElementById('ft-dataset-type').value,
    dataset_desc: document.getElementById('ft-desc').value,
    verified_data_path: document.getElementById('ft-data-path').value,
    base_model: document.getElementById('ft-base-model').value,
    lora_rank: parseInt(document.getElementById('ft-lora-rank').value),
    lora_alpha: parseInt(document.getElementById('ft-lora-alpha').value),
    batch_size: parseInt(document.getElementById('ft-batch').value),
    learning_rate: parseFloat(document.getElementById('ft-lr').value),
    epochs: parseInt(document.getElementById('ft-epochs').value),
    max_rows: parseInt(document.getElementById('ft-rows').value),
  };
}

async function startTraining() {
  const cfg = getConfig();
  document.getElementById('ft-start-btn').disabled = true;
  document.getElementById('ft-start-btn').textContent = 'Starting...';
  document.getElementById('ft-stop-btn').style.display = '';
  document.getElementById('train-empty').style.display = 'none';
  document.getElementById('train-metrics').style.display = '';
  lossData = [];

  try {
    const res = await fetch('/api/train/start', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(cfg),
    });
    if (!res.ok) { const e = await res.json(); alert('Error: '+(e.detail||'unknown')); resetTrainButtons(); return; }
    connectTrainingSSE();
  } catch(e) { alert('Error: '+e.message); resetTrainButtons(); }
}

function connectTrainingSSE() {
  if (trainingEventSource) trainingEventSource.close();
  trainingEventSource = new EventSource('/api/train/progress');

  trainingEventSource.onmessage = function(e) {
    try {
      const data = JSON.parse(e.data);
      if (data.event === 'stream_end') { trainingEventSource.close(); resetTrainButtons(); return; }
      updateTrainingUI(data);
    } catch(err) {}
  };

  trainingEventSource.onerror = function() {
    // Reconnect after 1s
    setTimeout(() => {
      if (trainingEventSource) trainingEventSource.close();
      // Check status
      fetch('/api/train/status').then(r=>r.json()).then(s => {
        if (s.status === 'running') connectTrainingSSE();
        else resetTrainButtons();
      });
    }, 1000);
  };
}

function updateTrainingUI(data) {
  const ev = data.event;
  document.getElementById('ft-phase').textContent = data.phase || (data.status||'');
  document.getElementById('ft-message').textContent = data.message || '';

  if (ev === 'progress' || ev === 'progress_delta') {
    const step = data.step || 0;
    const total = data.total_steps || 1;
    const pct = data.progress_percent || ((step/total)*100).toFixed(1);
    document.getElementById('train-progress').style.width = Math.min(pct, 100)+'%';
    document.getElementById('m-loss').textContent = data.loss != null ? data.loss.toFixed(4) : '—';
    document.getElementById('m-lr').textContent = data.lr != null ? data.lr.toExponential(2) : '—';
    document.getElementById('m-epoch').textContent = data.epoch != null ? data.epoch : '—';
    document.getElementById('m-eta').textContent = data.eta_seconds ? formatDuration(data.eta_seconds) : '—';
    document.getElementById('ft-phase').textContent = phase + '';

    if (data.loss != null) {
      lossData.push({ step: lossData.length, loss: data.loss });
      drawLossChart();
    }
  }

  if (ev === 'complete') {
    document.getElementById('train-progress').style.width = '100%';
    document.getElementById('ft-phase').textContent = 'Completed ✓';
    document.getElementById('ft-message').textContent = 'Final loss: '+(data.final_loss != null ? data.final_loss.toFixed(4) : '?');
    resetTrainButtons();
    loadModels();
    if (trainingEventSource) trainingEventSource.close();
  }

  if (ev === 'error') {
    document.getElementById('ft-phase').textContent = 'Error ✗';
    document.getElementById('ft-message').textContent = data.message || 'Unknown error';
    resetTrainButtons();
    if (trainingEventSource) trainingEventSource.close();
  }
}

async function stopTraining() {
  const save = confirm('Save checkpoint before stopping?');
  document.getElementById('ft-message').textContent = 'Stopping...';
  await fetch('/api/train/stop?save='+save, {method:'POST'});
  if (save) loadModels();
}

function resetTrainButtons() {
  document.getElementById('ft-start-btn').disabled = false;
  document.getElementById('ft-start-btn').textContent = 'Start Training';
  document.getElementById('ft-stop-btn').style.display = 'none';
}

async function exportModel() {
  const niche = document.getElementById('ft-niche').value;
  const adapterPath = 'models/adapters/'+niche;
  document.getElementById('ft-message').textContent = 'Exporting model...';
  await fetch('/api/export?niche='+encodeURIComponent(niche)+'&adapter_path='+encodeURIComponent(adapterPath), {method:'POST'});
  document.getElementById('ft-message').textContent = 'Export complete!';
  loadModels();
}

function formatDuration(seconds) {
  if (seconds < 60) return Math.round(seconds)+'s';
  if (seconds < 3600) return Math.floor(seconds/60)+'m '+Math.round(seconds%60)+'s';
  return Math.floor(seconds/3600)+'h '+Math.round((seconds%3600)/60)+'m';
}

// ── Loss Chart ──
function drawLossChart() {
  const canvas = document.getElementById('loss-chart');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  canvas.width = canvas.parentElement.clientWidth;
  canvas.height = canvas.parentElement.clientHeight;
  const w = canvas.width, h = canvas.height;
  const pad = {top:10, right:10, bottom:20, left:40};

  if (lossData.length < 2) {
    ctx.clearRect(0,0,w,h);
    ctx.fillStyle = '#8b949e';
    ctx.font = '13px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(lossData.length === 1 ? 'Loss: '+lossData[0].loss.toFixed(4) : 'Waiting for data...', w/2, h/2);
    return;
  }

  const vals = lossData.map(d => d.loss);
  const minLoss = Math.min(...vals) * 0.95;
  const maxLoss = Math.max(...vals) * 1.05;
  const range = maxLoss - minLoss || 1;

  const plotW = w - pad.left - pad.right;
  const plotH = h - pad.top - pad.bottom;

  ctx.clearRect(0,0,w,h);

  // Grid lines
  ctx.strokeStyle = '#1c2128';
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = pad.top + (plotH/4)*i;
    ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(w-pad.right, y); ctx.stroke();
    const val = maxLoss - (range/4)*i;
    ctx.fillStyle = '#8b949e'; ctx.font = '10px sans-serif';
    ctx.textAlign = 'right'; ctx.fillText(val.toFixed(2), pad.left-4, y+3);
  }

  // Loss line
  const step = Math.max(1, Math.floor(lossData.length / 200));
  const display = lossData.filter((_,i) => i % step === 0 || i === lossData.length-1);

  ctx.beginPath();
  ctx.strokeStyle = '#58a6ff';
  ctx.lineWidth = 2;
  display.forEach((d, i) => {
    const x = pad.left + (i / (display.length-1)) * plotW;
    const y = pad.top + (1 - (d.loss - minLoss) / range) * plotH;
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  });
  ctx.stroke();
}

// ── Docs ──
async function loadDocs() {
  try {
    const res = await fetch('/api/docs/v1');
    const d = await res.json();
    const body = document.getElementById('docs-body');
    body.innerHTML = renderDocs(d);
  } catch(e) { console.error('docs load error:', e); }
}

function renderDocs(d) {
  return `
    <div style="margin-bottom:6px;"><span class="docs-tag ml">v${d.version}</span>${d.name}</div>
    <div style="font-size:11px;margin-bottom:8px;">${d.description}</div>

    <h4>Architecture</h4>
    <ul>
      <li>Backend: <code>${d.architecture.backend}</code></li>
      <li>Training: <code>${d.architecture.training_isolation}</code></li>
      <li>Progress: <code>${d.architecture.progress_streaming}</code></li>
      <li>Framework: <code>${d.architecture.ml_framework}</code></li>
    </ul>

    <h4>Features</h4>
    <ul>
      <li><span class="docs-tag api">Chat</span>${d.features.chat.description}</li>
      <li><span class="docs-tag api">Train</span>${d.features.training.description}</li>
      <li><span class="docs-tag ml">Consensus</span>${d.features.consensus_verification.description}</li>
      <li><span class="docs-tag ml">Export</span>${d.features.export.format}</li>
    </ul>

    <h4>Endpoints</h4>
    <ul>
      <li><code>GET /api/models</code> — list models</li>
      <li><code>GET /api/chat</code> — chat with model</li>
      <li><code>POST /api/train/start</code> — start training</li>
      <li><code>GET /api/train/progress</code> — SSE progress stream</li>
      <li><code>GET /api/train/status</code> — current status</li>
      <li><code>POST /api/train/stop</code> — stop training</li>
      <li><code>GET /api/train/history</code> — past runs</li>
      <li><code>POST /api/export</code> — export model</li>
      <li><code>GET /api/leaderboard</code> — benchmark data</li>
      <li><code>GET /api/docs/v1</code> — this documentation</li>
    </ul>

    <h4>Data Flow</h4>
    <ol style="padding-left:14px;">
      <li>BigSet / local JSONL → raw dataset</li>
      <li>4 commandcode models verify each row via consensus</li>
      <li>Verified data → train/valid split</li>
      <li>MLX-LoRA subprocess fine-tunes, streams progress via SSE</li>
      <li>Adapter merged → GGUF → Ollama → model dropdown</li>
    </ol>

    <h4>Hardware</h4>
    <ul>
      <li>Apple M5 Max + 128GB unified memory</li>
      <li>MLX Metal: available</li>
      <li>Fits 70B 4-bit model + TurboVec index + cache</li>
    </ul>

    <h4>Pipeline Modules</h4>
    <ul>
      <li><code>pipeline/consensus_verifier.py</code></li>
      <li><code>pipeline/training_worker.py</code></li>
      <li><code>pipeline/training_manager.py</code></li>
      <li><code>pipeline/eval_harness.py</code></li>
      <li><code>pipeline/export_gguf.py</code></li>
      <li><code>pipeline/recursive_loop.py</code></li>
    </ul>

    <h4>Dependencies</h4>
    <ul>
      ${Object.entries(d.external_dependencies).map(([k,v]) => `<li><code>${k}</code> — ${v}</li>`).join('')}
    </ul>

    <div style="margin-top:8px;font-size:10px;color:var(--text-secondary);border-top:1px solid var(--border);padding-top:6px;">
      Port: ${d.ui.port} · Taste: ${d.taste_configuration.ports}
    </div>
  `;
}

function toggleDocs() {
  const body = document.getElementById('docs-body');
  const arrow = document.getElementById('docs-arrow');
  const open = body.classList.toggle('open');
  arrow.classList.toggle('open', open);
}

// ── Init ──
loadModels();
loadLeaderboard();
loadDocs();
setInterval(loadLeaderboard, 30000);
</script>
</body>
</html>
"""


# ── Main ───────────────────────────────────────────────────

def main():
    port = int(os.environ.get("PORT", 7100))
    print(f"Starting Fine-Tuning Platform on http://localhost:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")


if __name__ == "__main__":
    main()
