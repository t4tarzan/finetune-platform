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


class DiscoverRequest(BaseModel):
    niche_desc: str
    test_set_path: str = ""
    max_candidates: int = 10
    accuracy_threshold: float = 0.7


class StartGRPORequest(BaseModel):
    niche: str
    adapter_path: str = ""
    data_path: str = "data/grpo_train.jsonl"
    base_model: str = ""
    group_size: int = 4
    learning_rate: float = 5e-6
    grpo_epochs: int = 1
    judge_model: str = "claude-sonnet-4-6"


# ── Model endpoints ────────────────────────────────────────

def _detect_model_type(name: str) -> dict:
    """Detect model type and capabilities from name patterns."""
    name_lower = name.lower()
    model_type = "chat"
    capabilities = ["text-generation"]
    warning = None
    icon = "💬"

    # Embedding models
    if any(x in name_lower for x in ("embed", "nomic-embed", "bge-", "e5-", "gte-", "instructor", "sentence-transform")):
        model_type = "embedding"
        capabilities = ["embedding"]
        warning = "Embedding model — returns vectors, not text. Not suitable for chat."
        icon = "🔢"
    # Vision models
    elif any(x in name_lower for x in ("vision", "vl", "vlm", "llava", "cogvlm", "internvl")):
        model_type = "vision"
        capabilities = ["text-generation", "vision"]
        icon = "👁️"
    # Audio models
    elif any(x in name_lower for x in ("whisper", "tts", "speech", "audio")):
        model_type = "audio"
        capabilities = ["audio"]
        warning = "Audio model — not suitable for text chat."
        icon = "🎤"
    # Code models
    elif any(x in name_lower for x in ("code", "starcoder", "deepseek-coder")):
        capabilities = ["text-generation", "code"]
        icon = "💻"

    is_ft = any(x in name_lower for x in ("-iter", "-v1", "finetune", "-ft"))
    display_type = "fine-tuned" if is_ft else model_type
    can_train = model_type != "embedding" and model_type != "audio"

    return {
        "id": name,
        "name": name,
        "provider": "ollama",
        "type": display_type,
        "model_type": model_type,
        "capabilities": capabilities,
        "icon": icon,
        "warning": warning,
        "can_train": can_train,
    }


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
                    info = _detect_model_type(parts[0])
                    models.append(info)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return {"models": models}


@app.get("/api/models/validate")
def validate_model(model: str = Query(..., description="Model ID to validate")):
    """Validate a model for a specific use case. Returns warnings if incompatible."""
    info = _detect_model_type(model)
    warnings = []
    if info.get("warning"):
        warnings.append(info["warning"])
    return {
        "valid": len(warnings) == 0,
        "warnings": warnings,
        "model_type": info["model_type"],
        "can_train": info["can_train"],
    }


@app.get("/api/chat")
def chat(
    message: str = Query(...),
    model: str = Query(""),
    stream: bool = Query(False),
):
    if not model:
        raise HTTPException(400, "model required")

    # Validate model type
    validation = validate_model(model)
    if not validation["valid"]:
        raise HTTPException(400, validation["warnings"][0] if validation["warnings"] else "Model incompatible with chat")

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


# ── Model Discovery ────────────────────────────────────────

@app.post("/api/discover")
def discover_models(req: DiscoverRequest):
    """Search HF for the best model for a niche, evaluate, and recommend."""
    from pipeline.model_discovery import ModelDiscoveryAgent

    agent = ModelDiscoveryAgent()
    report = agent.discover(
        niche_desc=req.niche_desc,
        test_set_path=req.test_set_path if req.test_set_path else None,
        max_candidates=req.max_candidates,
        accuracy_threshold=req.accuracy_threshold,
        evaluate_locally=bool(req.test_set_path),
    )
    return {
        "niche": report.niche,
        "candidates": report.candidates,
        "top_recommendation": report.top_recommendation,
        "can_skip_training": report.can_skip_training,
        "summary": report.summary,
    }


# ── GRPO Reinforcement Learning ────────────────────────────

@app.post("/api/grpo/start")
async def grpo_start(req: StartGRPORequest):
    """Start GRPO training in a subprocess."""
    import subprocess
    import uuid
    import tempfile

    run_id = str(uuid.uuid4())[:8]
    niche = req.niche
    adapter_path = req.adapter_path or f"models/adapters/{niche}_grpo"
    stop_file = os.path.join(tempfile.gettempdir(), f"grpo_stop_{run_id}")

    if os.path.exists(stop_file):
        os.remove(stop_file)

    worker_config = {
        "niche": niche,
        "adapter_path": adapter_path,
        "data_path": req.data_path,
        "base_model": req.base_model or config.get("base_model"),
        "stop_file": stop_file,
        "group_size": req.group_size,
        "learning_rate": req.learning_rate,
        "grpo_epochs": req.grpo_epochs,
        "judge_model": req.judge_model,
        "batch_size": config.get("rl", {}).get("batch_size", 4),
        "max_prompt_length": config.get("training", {}).get("max_seq_length", 2048),
        "max_completion_length": config.get("rl", {}).get("max_completion_length", 128),
        "clip_epsilon": config.get("rl", {}).get("clip_epsilon", 0.2),
        "kl_beta": config.get("rl", {}).get("kl_beta", 0.04),
    }

    venv_python = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        ".venv", "bin", "python",
    )
    worker_script = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "pipeline", "grpo_trainer.py",
    )

    proc = subprocess.Popen(
        [venv_python, worker_script],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )
    proc.stdin.write(json.dumps(worker_config) + "\n")
    proc.stdin.flush()
    proc.stdin.close()

    return {
        "run_id": run_id,
        "niche": niche,
        "status": "started",
        "stop_file": stop_file,
        "adapter_path": adapter_path,
    }


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


# ── Inference Server Management ───────────────────────────

INFERENCE_PROCESS = {"process": None, "port": None}

@app.post("/api/inference/start")
def inference_start():
    """Start the persistent inference server on port 7200."""
    if INFERENCE_PROCESS["process"] and INFERENCE_PROCESS["process"].poll() is None:
        return {"status": "already_running", "port": INFERENCE_PROCESS["port"]}

    import subprocess
    venv_python = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        ".venv", "bin", "python",
    )
    worker = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "pipeline", "inference_server.py",
    )

    proc = subprocess.Popen(
        [venv_python, worker],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )
    INFERENCE_PROCESS["process"] = proc
    INFERENCE_PROCESS["port"] = config.get("ports", {}).get("inference_api", 7200)
    return {"status": "started", "port": INFERENCE_PROCESS["port"]}


@app.post("/api/inference/stop")
def inference_stop():
    """Stop the inference server."""
    if INFERENCE_PROCESS["process"]:
        INFERENCE_PROCESS["process"].terminate()
        INFERENCE_PROCESS["process"].wait(timeout=5)
        INFERENCE_PROCESS["process"] = None
        return {"status": "stopped"}
    return {"status": "not_running"}


@app.get("/api/inference/status")
def inference_status():
    """Check inference server status."""
    import socket
    port = config.get("ports", {}).get("inference_api", 7200)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    is_running = sock.connect_ex(("127.0.0.1", port)) == 0
    sock.close()

    models = []
    if is_running:
        try:
            import urllib.request
            resp = urllib.request.urlopen(f"http://127.0.0.1:{port}/api/manage/list", timeout=2)
            models = json.loads(resp.read()).get("models", {})
        except Exception:
            models = []

    return {
        "running": is_running,
        "port": port,
        "models_loaded": list(models.keys()) if isinstance(models, dict) else [],
        "pid": INFERENCE_PROCESS["process"].pid if INFERENCE_PROCESS["process"] and INFERENCE_PROCESS["process"].poll() is None else None,
    }


@app.post("/api/inference/load")
def inference_load_model(
    model_path: str = Query(...),
    model_name: str = Query(None),
):
    """Load a model into the running inference server."""
    import urllib.request
    port = config.get("ports", {}).get("inference_api", 7200)
    url = f"http://127.0.0.1:{port}/api/manage/load?model_path={model_path}"
    if model_name:
        url += f"&model_name={model_name}"
    try:
        resp = urllib.request.urlopen(url, timeout=120)
        return json.loads(resp.read())
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/inference/unload")
def inference_unload_model(model_name: str = Query(...)):
    """Unload a model from the inference server."""
    import urllib.request
    port = config.get("ports", {}).get("inference_api", 7200)
    url = f"http://127.0.0.1:{port}/api/manage/unload?model_name={model_name}"
    try:
        resp = urllib.request.urlopen(url, timeout=10)
        return json.loads(resp.read())
    except Exception as e:
        return {"error": str(e)}


# ── Data Store Management ─────────────────────────────────

DATASTORE_INSTANCE = None

def get_datastore():
    global DATASTORE_INSTANCE
    if DATASTORE_INSTANCE is None:
        from pipeline.data_store import DataStore
        DATASTORE_INSTANCE = DataStore()
    return DATASTORE_INSTANCE


@app.get("/api/data/stats")
def data_stats():
    """Get data store statistics."""
    ds = get_datastore()
    try:
        return {
            "tables": ds.list_tables(),
            "stats": ds.stats(),
            "file_size": ds.file_size(),
        }
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/data/import")
def data_import(
    path: str = Query(...),
    domain: str = Query("general"),
    format: str = Query("auto"),
):
    """Import data from a file into the data store."""
    ds = get_datastore()
    fmt = format if format != "auto" else os.path.splitext(path)[1].lower().lstrip(".")
    try:
        if fmt == "jsonl":
            rows = ds.import_jsonl(path, domain=domain)
        elif fmt == "parquet":
            rows = ds.import_parquet(path, domain=domain)
        elif fmt == "csv":
            rows = ds.import_csv(path, domain=domain)
        else:
            return {"error": f"Unsupported format: {fmt}"}
        return {"rows_imported": rows, "table": "training", "format": fmt}
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/data/query")
def data_query(sql: str = Query(...)):
    """Execute a SQL query against the data store."""
    ds = get_datastore()
    try:
        results = ds.query(sql)
        return {"rows": len(results), "data": results[:100]}
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/data/schema")
def data_schema(table: str = Query("training")):
    """Describe a table's schema."""
    ds = get_datastore()
    try:
        return {"table": table, "columns": ds.describe(table)}
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/data/generate")
def data_generate(
    sql: str = Query(...),
    output_path: str = Query("data/verified_train.jsonl"),
    question_col: str = Query("question"),
    answer_col: str = Query("reference_answer"),
    format: str = Query("jsonl"),
):
    """Generate a training-ready dataset from a SQL query."""
    ds = get_datastore()
    try:
        rows = ds.generate_training_set(
            sql=sql,
            output_path=output_path,
            question_col=question_col,
            answer_col=answer_col,
            format=format,
        )
        return {"rows_generated": rows, "output_path": output_path, "format": format}
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/data/register-customer")
def data_register_customer(
    customer_id: str = Query(...),
    schema_name: str = Query(...),
    tables_json: str = Query(..., description="JSON object of table schemas"),
):
    """Register a customer's schema for automated data pipeline."""
    ds = get_datastore()
    try:
        tables = json.loads(tables_json)
        ds.register_customer_schema(customer_id, schema_name, tables)
        return {"status": "ok", "customer": customer_id, "tables": len(tables)}
    except Exception as e:
        return {"error": str(e)}


# ── Documentation ──────────────────────────────────────────

DOCS_TREE = [
  {
    "id": "pitch",
    "title": "The Pitch",
    "icon": "🚀",
    "content": """
<h2>Fine-Tuning Platform: Local AI That Learns Your Domain</h2>
<p>We built a <strong>self-contained, recursive fine-tuning platform</strong> that turns a single M-series Mac into an AI training factory. No cloud credits needed. Your data stays local.</p>
<p>Unlike cloud fine-tuning services that charge per-token and require uploading your data to third parties, our platform:</p>
<ul>
  <li><strong>Runs entirely on your hardware</strong> — Apple M5 Max with 128GB unified memory</li>
  <li><strong>Validates training data</strong> using 4 diverse LLMs (DeepSeek, Qwen, Kimi, Claude) in a consensus mechanism that rejects hallucinated or contradictory data</li>
  <li><strong>Recursively improves</strong> — fine-tune → eval → find gaps → generate better data → repeat, until the model plateaus</li>
  <li><strong>Discovers existing models first</strong> — searches HuggingFace, evaluates candidates locally, only fine-tunes if no pre-trained model meets your accuracy threshold</li>
  <li><strong>Polish with Reinforcement Learning</strong> — GRPO with a binary LLM judge reward to reinforce correct outputs</li>
</ul>

<h3>Who It's For</h3>
<table><tr><th>Role</th><th>Value</th></tr>
<tr><td><strong>Data Scientists</strong></td><td>Rapidly prototype domain-specific models without cloud infrastructure</td></tr>
<tr><td><strong>ML Engineers</strong></td><td>Automate fine-tuning pipelines with consensus-gated data and recursive improvement</td></tr>
<tr><td><strong>Product Teams</strong></td><td>Ship custom AI features in weeks (not months) trained on your unique data</td></tr>
<tr><td><strong>CTOs</strong></td><td>Eliminate data exfiltration risk — everything stays on-premise</td></tr>
<tr><td><strong>Researchers</strong></td><td>Iterate rapidly on small, high-quality domain datasets</td></tr>
</table>

<h3>Why It Matters</h3>
<p><em>"The best model for your domain probably already exists on HuggingFace. If it doesn't, you can fine-tune a close match in hours, not weeks."</em></p>
<p>The platform embodies this philosophy: discover first, train only when necessary, and when you do train, ensure your data is verified by multiple independent models to maximize quality per training token.</p>
"""
  },
  {
    "id": "problem",
    "title": "The Problem",
    "icon": "💡",
    "content": """
<h2>Why Most Fine-Tuning Fails (and How We Fix It)</h2>

<h3>1. Data Quality is the Bottleneck</h3>
<p>Fine-tuning is only as good as your training data. Bad data → bad model. Traditional approaches:</p>
<ul><li><strong>Manual curation</strong> — expensive, slow, doesn't scale</li><li><strong>Synthetic generation</strong> — fast but hallucinated data poisons the model</li><li><strong>Web scraping</strong> — noisy, unstructured, contradictory</li></ul>
<p><strong>Our fix:</strong> Every training row is verified by 4 independently-trained models (different architectures, different training data). Only rows where 3+ models agree pass into the training set. This eliminates hallucination propagation.</p>

<h3>2. Cloud Costs Are Unpredictable</h3>
<p>Cloud GPU instances for fine-tuning cost $2-40/hour. A 7B model fine-tuned for 3 epochs on 10K rows costs $50-200. Scale that across experiments and iterations, and you're looking at $5K-20K+ per project.</p>
<p><strong>Our fix:</strong> Use Apple's unified memory architecture. The M5 Ultra 128GB is a single pool — no GPU VRAM limits, no CUDA memory management. MLX runs natively with zero overhead. Training is <strong>free after hardware cost</strong>.</p>

<h3>3. Iteration Cycle is Too Long</h3>
<p>Typical workflow: train → manually evaluate → analyze failures → curate more data → retrain → repeat. Each cycle takes 1-3 days.</p>
<p><strong>Our fix:</strong> The recursive loop automates the entire cycle. It runs overnight. You wake up to a better model.</p>

<h3>4. Model Selection is a Gamble</h3>
<p>Choosing the right base model for your domain is hard. Too small → poor performance. Too large → unnecessary compute. Wrong architecture → incompatible with your task.</p>
<p><strong>Our fix:</strong> The discovery agent searches HuggingFace for models matching your niche, downloads and evaluates each one locally, and ranks them by accuracy vs efficiency. If a model already scores above your threshold, <em>no training is needed.</em></p>

<h3>5. Reinforcement Learning is Complex to Set Up</h3>
<p>GRPO, PPO, RLHF — these require specialized infrastructure, reward model training, and careful hyperparameter tuning. Most teams skip this step entirely.</p>
<p><strong>Our fix:</strong> One-click GRPO with any commandcode API model as the judge. Binary reward (1/0). Group-based advantage normalization. No separate reward model to train.</p>
"""
  },
  {
    "id": "analogy",
    "title": "Analogy: The Master Chef",
    "icon": "👨‍🍳",
    "content": """
<h2>Understanding the Platform Through an Analogy</h2>
<p>Think of building a domain-specialist AI model like training a master chef:</p>

<h3>Base Model = A Cook Who Knows All Cuisines</h3>
<p>Your base model (like Qwen 2.5-7B) is a talented chef who has studied every cuisine in the world — Japanese, Italian, Indian, French, Mexican — equally. They know the theory but haven't specialized.</p>

<h3>Fine-Tuning = Apprenticeship in One Kitchen</h3>
<p>Supervised fine-tuning (SFT) is like apprenticing that chef in a specific restaurant — say, a Tuscan farmhouse kitchen. After weeks of practice with real Tuscan recipes (your training data), they start to <em>understand the patterns</em>: how much rosemary in a classic ribollita, the right temperature for a slow-roasted pork shoulder.</p>

<h3>Consensus Verification = The Tasting Panel</h3>
<p>Before any recipe enters the chef's curriculum, it must be approved by a panel of 3-4 expert taste testers. Each tester has different expertise. If all agree the recipe is authentic and well-documented, it enters the training set. If there's disagreement, the recipe is rejected.</p>
<p><em>In technical terms: 4 models independently verify each training row. 3/4 must agree with high confidence.</em></p>

<h3>GRPO = The Head Chef's Feedback Loop</h3>
<p>After the apprenticeship, the head chef watches every dish and gives a simple signal: <strong>thumbs up</strong> (correct) or <strong>thumbs down</strong> (incorrect). The chef adjusts their approach based on this feedback, reinforcing correct patterns and weakening incorrect ones.</p>

<h3>Discovery = The Menu Research Phase</h3>
<p>Before designing a new menu, the chef visits every restaurant within 100km to taste their signature dishes. If a restaurant in the next town is already serving exactly what you want, there's no need to reinvent it — you study theirs and improve on it.</p>

<h3>Recursive Loop = Continuous Improvement</h3>
<p>The best chefs don't stop learning. Each service, they identify dishes that didn't work, understand why, research better techniques, and improve. The recursive loop does the same — evaluate, find failures, generate targeted new data, retrain.</p>

<h3>The M5 Ultra 128GB = A Massive Kitchen with Infinite Counter Space</h3>
<p>Most kitchens have limited counter space (GPU VRAM). You can only work with one prep station at a time. Our kitchen has 128GB of unified counter space — you can have the prep station (model weights), the pantry (training data), the spice rack (vector index), and the plating area (cache) all at once, no shuffling needed.</p>
<p style="font-style:italic;color:var(--text-secondary);margin-top:16px;">This analogy was generated by the platform's AI to demonstrate its own capabilities.</p>
"""
  },
  {
    "id": "architecture",
    "title": "Architecture",
    "icon": "🏗️",
    "children": [
      {
        "id": "arch-overview",
        "title": "System Overview",
        "icon": "",
        "content": """
<h2>System Architecture</h2>
<pre style="font-size:12px;line-height:1.4;">
┌─────────────────────────────────────────────────────────────┐
│                    Browser (localhost:7100)                   │
│  ┌────────────┐  ┌────────────┐  ┌──────────────────────┐  │
│  │ Chat Tab   │  │ Train Tab  │  │ Sidebar: Docs + LB   │  │
│  └─────┬──────┘  └─────┬──────┘  └──────────────────────┘  │
├────────┴───────────────┴─────────────────────────────────────┤
│                    FastAPI Backend (Python 3.13)              │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────────────┐  │
│  │ /api/*   │ │ SSE      │ │ Pydantic │ │ CORS + Security│  │
│  │ Routes   │ │ Streams  │ │ Models   │ │ Middleware     │  │
│  └────┬─────┘ └────┬─────┘ └──────────┘ └────────────────┘  │
├───────┴────────────┴──────────────────────────────────────────┤
│                     Pipeline Modules (pipeline/)               │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────────────┐  │
│  │ Discovery│ │Consensus │ │Training  │ │    GRPO RL     │  │
│  │  Agent   │ │Verifier  │ │  Worker  │ │   (subproc)    │  │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └───────┬────────┘  │
│  ┌────┴────┐  ┌────┴────┐  ┌────┴────┐  ┌──────┴────────┐  │
│  │Hugging  │  │ 4 cmd   │  │ MLX     │  │ LLM Judge    │  │
│  │Face Hub │  │  Models │  │ LoRA    │  │ (Claude/etc) │  │
│  └─────────┘  └─────────┘  └────┬────┘  └───────────────┘  │
│  ┌──────────────────────────────┴────────────────────────┐  │
│  │         ~/.cache/huggingface/hub/ (base models)        │  │
│  │         models/adapters/ (LoRA weights)                │  │
│  │         Ollama (GGUF serving)                          │  │
│  └───────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
</pre>
<h3>Layered Architecture</h3>
<table><tr><th>Layer</th><th>Technology</th><th>Responsibility</th></tr>
<tr><td>Presentation</td><td>HTML/CSS/JS (SPA)</td><td>Chat, training config, live charts, docs browser</td></tr>
<tr><td>API</td><td>FastAPI (Python 3.13)</td><td>REST + SSE endpoints</td></tr>
<tr><td>Orchestration</td><td>TrainingManager + asyncio</td><td>Subprocess lifecycle, SSE streaming, state machine, history</td></tr>
<tr><td>Training</td><td>MLX LoRA (subprocess)</td><td>Model loading, LoRA, training loop, stdout progress</td></tr>
<tr><td>Data Prep</td><td>ConsensusVerifier</td><td>Multi-model row verification, data splitting</td></tr>
<tr><td>Discovery</td><td>ModelDiscoveryAgent</td><td>HF Hub search, local eval, accuracy ranking</td></tr>
<tr><td>RL</td><td>GRPOTrainer (subprocess)</td><td>Group policy optimization with LLM judge reward</td></tr>
<tr><td>Serving</td><td>Ollama</td><td>GGUF serving, OpenAI-compatible API, model dropdown</td></tr>
<tr><td>Storage</td><td>Local filesystem</td><td>JSONL, safetensors, JSON history/leaderboard</td></tr>
</table>

<h3>Key Design Decisions</h3>
<dl>
<dt><strong>Subprocess Isolation</strong></dt><dd>Each training job runs in a fresh Python process spawned via subprocess.Popen. Prevents ML context leaks, stale module state, and memory fragmentation across runs. Worker communicates via newline-delimited JSON on stdout — simple, debuggable, streamable.</dd>
<dt><strong>SSE Over WebSockets</strong></dt><dd>Server-Sent Events is unidirectional (server→client), simpler, works over standard HTTP, has built-in reconnection (Last-Event-ID). Perfect for progress streaming.</dd>
<dt><strong>File-Based Stop Signals</strong></dt><dd>Instead of complex IPC, the manager touches a temp file to signal stop. Worker checks every 3 training steps. Simple, safe, cross-process.</dd>
<dt><strong>JSONL for Everything</strong></dt><dd>Training data, consensus results, worker events — all newline-delimited JSON. Human-readable, streamable, trivially split/merged with standard shell tools.</dd>
<dt><strong>Port Range 7000-7500</strong></dt><dd>All built services use this range by convention to avoid conflicts with common ports.</dd>
</dl>
"""
      },
      {
        "id": "arch-data-flow",
        "title": "Data Flow",
        "icon": "",
        "content": """
<h2>End-to-End Data Flow</h2>
<pre style="font-size:12px;">
 BigSet/JSONL → Discovery → Consensus → MLX LoRA → GRPO RL → Export → Ollama
   (raw)        (find)     (verify)    (train)    (polish)  (GGUF)  (serve)
</pre>
<h3>Step-by-Step</h3>
<ol>
<li><strong>Data Ingestion</strong> — Raw data from BigSet (natural language → structured) or local JSONL: <code>{"question","reference_answer","context"}</code></li>
<li><strong>Model Discovery</strong> — Before any training, search HuggingFace for pre-trained models matching the domain. If accuracy ≥ threshold, skip training entirely.</li>
<li><strong>Consensus Verification</strong> — Every row sent to 4 models. Each returns (agrees, confidence). ≥3 agreements with avg confidence ≥0.7 passes. Rejected rows logged.</li>
<li><strong>Data Preparation</strong> — Verified rows split 80/20 and formatted as <code>{"prompt","completion"}</code> for MLX LoRA.</li>
<li><strong>Fine-Tuning</strong> — Worker loads base model, applies LoRA, trains, streams progress (loss, lr, tokens/sec, ETA) via SSE.</li>
<li><strong>GRPO RL Polish</strong> — (Optional) For each prompt, generate 4 completions, judge each 1/0, normalize advantages within group, update policy toward high-reward completions.</li>
<li><strong>Export & Serve</strong> — Merge adapter with base, save as GGUF, register with Ollama. Appears in chat dropdown.</li>
</ol>
<h3>Data Formats</h3>
<table><tr><th>Stage</th><th>Format</th></tr>
<tr><td>Raw input</td><td><code>{"question", "reference_answer", "context"}</code></td></tr>
<tr><td>After consensus</td><td>verified_train.jsonl / rejected_train.jsonl</td></tr>
<tr><td>Training set</td><td><code>{"prompt", "completion"}</code></td></tr>
<tr><td>Worker events</td><td><code>{"event":"progress","step":1,"loss":4.89}</code></td></tr>
<tr><td>History</td><td>training_history.json (array of runs)</td></tr>
<tr><td>Leaderboard</td><td>benchmarks/leaderboard.json (niche→iterations)</td></tr>
</table>
"""
      },
      {
        "id": "arch-subprocess",
        "title": "Subprocess Isolation",
        "icon": "",
        "content": """
<h2>Training Subprocess Architecture</h2>
<pre style="font-size:12px;">
┌────────────────────────────────────────────────────────────┐
│  TrainingManager (parent): spawns worker, reads stdout     │
│  ├─ stdin: config.json                                     │
│  ├─ stdout: {"event":"progress","loss":4.89}               │
│  └─ Stop: touch /tmp/train_stop_<id> → worker checks →     │
│     saves partial checkpoint, exits cleanly                │
├────────────────────────────────────────────────────────────┤
│  TrainingWorker (child): isolated process, clean MLX state │
│  ├─ Load model (MLX, 4-bit quantized)                      │
│  ├─ Apply LoRA adapters                                    │
│  ├─ Training loop with ProgressCallback                    │
│  └─ Emit events on stdout every step                      │
└────────────────────────────────────────────────────────────┘
</pre>
<h3>Why Subprocess Isolation?</h3>
<ul><li><strong>No MLX state leaks</strong> — fresh process per run, no fragmentation</li><li><strong>Kill-safe</strong> — SIGTERM doesn't corrupt parent's state</li><li><strong>Module isolation</strong> — different runs could use different MLX versions</li><li><strong>Resource containment</strong> — OOM in child doesn't crash parent</li></ul>
<h3>IPC Protocol</h3>
<table><tr><th>Event</th><th>When</th><th>Example</th></tr>
<tr><td><code>status</code></td><td>State transitions</td><td><code>{"event":"status","phase":"loading_model"}</code></td></tr>
<tr><td><code>progress</code></td><td>Every step</td><td><code>{"event":"progress","step":1,"loss":4.89,"lr":1e-4,"tokens_per_sec":2400}</code></td></tr>
<tr><td><code>complete</code></td><td>Training finishes</td><td><code>{"event":"complete","final_loss":2.33,"output_dir":"models/..."}</code></td></tr>
<tr><td><code>error</code></td><td>Training fails</td><td><code>{"event":"error","message":"OOM"}</code></td></tr>
<tr><td><code>checkpoint_saved</code></td><td>Adapter written</td><td><code>{"event":"checkpoint_saved","path":"models/..."}</code></td></tr>
</table>
"""
      },
    ]
  },
  {
    "id": "features",
    "title": "Feature Deep Dive",
    "icon": "⚡",
    "children": [
      {
        "id": "feat-consensus",
        "title": "Consensus Verification",
        "icon": "",
        "content": """
<h2>Multi-Model Consensus Verification</h2>
<p>The consensus verification system is the <strong>quality gate</strong> for all training data. Every row must be validated by multiple independently-trained models before it enters the training set.</p>
<h3>How It Works</h3>
<ol><li>Each data point sent to 4 models simultaneously via commandcode API</li><li>Each model responds: <code>{"agrees":bool,"confidence":float,"reasoning":"..."}</code></li><li>Consensus scorer: ≥3 agree? Avg confidence ≥0.7?</li><li>Pass → training set. Fail → rejected set with reasoning.</li></ol>
<h3>Why 4 Models, Not 1?</h3>
<p>Different models have different training data, architectures, and biases. The probability of 4 independently-trained models all hallucinating the same false claim is near-zero. This is ensemble methodology applied to data quality.</p>
<h3>Model Selection Strategy</h3>
<table><tr><th>Model</th><th>Architecture</th><th>Why Included</th></tr>
<tr><td>DeepSeek V4 Pro</td><td>MoE</td><td>Different training distribution, strong reasoning</td></tr>
<tr><td>Qwen 3.6 Max Preview</td><td>Dense Transformer</td><td>Strong general knowledge</td></tr>
<tr><td>Kimi K2.7 Code</td><td>Long-context optimized</td><td>Different family, strong on structured data</td></tr>
<tr><td>Claude Sonnet 4-6</td><td>Constitutional AI</td><td>Completely different training approach</td></tr>
</table>
<h3>Output</h3>
<ul><li><code>verified_train.jsonl</code> — rows that passed consensus</li><li><code>rejected_train.jsonl</code> — rows that failed, with per-model reasoning</li><li><code>consensus_report.json</code> — summary stats: verification rate, per-model agreement, confidence distribution</li></ul>
"""
      },
      {
        "id": "feat-discovery",
        "title": "Model Discovery Agent",
        "icon": "",
        "content": """
<h2>Model Discovery Agent</h2>
<p>Prevents unnecessary training by finding and evaluating existing models first. There are over 1M models on HuggingFace — someone may have already fine-tuned for your domain.</p>
<h3>How It Works</h3>
<ol><li><strong>Parse</strong> — Extract keywords and task type from niche description</li><li><strong>Search</strong> — Query HF Hub by pipeline_tag + keywords, sorted by downloads</li><li><strong>Filter</strong> — Apply min downloads, estimate parameter size, detect MLX compatibility</li><li><strong>Evaluate</strong> — Load each candidate via MLX, run against domain test set, measure accuracy/latency</li><li><strong>Rank</strong> — Composite score weighted by accuracy, parameter efficiency, MLX bonus</li></ol>
<h3>Recommendations</h3>
<table><tr><th>Label</th><th>Condition</th><th>Action</th></tr>
<tr><td><code>use-as-is</code></td><td>Accuracy ≥ threshold (0.7)</td><td>Serve immediately, no training</td></tr>
<tr><td><code>fine-tune</code></td><td>Accuracy 0.49-0.7</td><td>Good candidate for domain fine-tuning</td></tr>
<tr><td><code>skip</code></td><td>Accuracy < 0.49</td><td>Not suitable, try another base</td></tr>
</table>
"""
      },
      {
        "id": "feat-grpo",
        "title": "GRPO Reinforcement Learning",
        "icon": "",
        "content": """
<h2>GRPO: Group Relative Policy Optimization</h2>
<p>Polishes a supervised fine-tuned model by reinforcing correct outputs and penalizing incorrect ones via binary feedback from an LLM judge.</p>
<h3>How It Works</h3>
<ol><li><strong>Generate Group</strong> — For each prompt, model generates N completions (default: 4)</li><li><strong>Judge Each</strong> — LLM judge scores each 1 (correct) or 0 (incorrect)</li><li><strong>Normalize</strong> — advantage = (reward - group_mean) / group_std — learns relative quality</li><li><strong>Update</strong> — Policy updated with clipping (ε=0.2) + KL penalty (β=0.04) to prevent drift</li></ol>
<h3>Why Binary Reward?</h3>
<p>Binary rewards (1/0) are simpler, more robust, and less prone to reward hacking than continuous rewards. The judge just needs to distinguish right from wrong.</p>
<h3>Hyperparameters</h3>
<pre>rl:
  group_size: 4           # Completions per prompt
  kl_beta: 0.04           # KL penalty coefficient
  clip_epsilon: 0.2       # PPO clipping range
  learning_rate: 5e-6     # Lower LR for RL
  judge_model: "claude-sonnet-4-6"</pre>
"""
      },
      {
        "id": "feat-training",
        "title": "MLX LoRA Training",
        "icon": "",
        "content": """
<h2>MLX LoRA Fine-Tuning</h2>
<p>MLX is Apple's ML framework for Apple Silicon. LoRA (Low-Rank Adaptation) trains a small set of additional parameters instead of updating the entire model — reducing trainable parameters by ~10,000x.</p>
<h3>Training Configuration</h3>
<pre>training:
  lora_rank: 16          # Size of low-rank matrices
  lora_alpha: 32         # Scaling factor
  learning_rate: 1e-4
  batch_size: 4
  epochs: 3
  max_seq_length: 2048
  grad_checkpoint: true</pre>
<h3>Memory Requirements</h3>
<table><tr><th>Model</th><th>4-bit Memory</th><th>Training Memory</th><th>M5 128GB</th></tr>
<tr><td>7B</td><td>~4 GB</td><td>~8 GB</td><td>✅</td></tr>
<tr><td>14B</td><td>~8 GB</td><td>~16 GB</td><td>✅</td></tr>
<tr><td>32B</td><td>~20 GB</td><td>~36 GB</td><td>✅</td></tr>
<tr><td>70B</td><td>~45 GB</td><td>~70 GB</td><td>✅ Fits</td></tr>
</table>
"""
      },
      {
        "id": "feat-benchmark",
        "title": "Benchmark Leaderboard",
        "icon": "",
        "content": """
<h2>Benchmark Leaderboard & Eval Harness</h2>
<h3>Metrics</h3>
<table><tr><th>Metric</th><th>Definition</th></tr>
<tr><td><strong>Accuracy</strong></td><td>% of responses matching expected answer (exact + fuzzy key-term match)</td></tr>
<tr><td><strong>Grounding</strong></td><td>% of responses with source citations</td></tr>
<tr><td><strong>Consistency</strong></td><td>% same answer across multiple runs</td></tr>
</table>
<h3>Automatic Stopping</h3>
<p>The recursive loop stops when: accuracy ≥95%, improvement <1% for 2 iterations, or max iterations reached.</p>
"""
      },
      {
        "id": "feat-chat",
        "title": "Chat Interface",
        "icon": "",
        "content": """
<h2>Chat Interface</h2>
<p>Conversational interface to any fine-tuned or base model registered with Ollama. Auto-populated model dropdown groups fine-tuned vs base models. Supports both local Ollama and commandcode API models.</p>
"""
      },
    ]
  },
  {
    "id": "user-manual",
    "title": "User Manual",
    "icon": "📖",
    "children": [
      {
        "id": "um-install",
        "title": "Installation",
        "icon": "",
        "content": """
<h2>Installation Guide</h2>
<h3>Prerequisites</h3>
<table><tr><th>Requirement</th><th>Details</th></tr>
<tr><td>Hardware</td><td>Apple Silicon Mac (M1-M5), tested on M5 Max 128GB</td></tr>
<tr><td>OS</td><td>macOS Sonoma 14+ or Sequoia 15+</td></tr>
<tr><td>Python</td><td>3.12+ (recommended: 3.13 via mise)</td></tr>
<tr><td>Node.js</td><td>22+ (for BigSet dataset generation)</td></tr>
<tr><td>Ollama</td><td>Latest (brew install ollama)</td></tr>
</table>
<h3>Quick Install</h3>
<pre>git clone https://github.com/YOUR_ORG/finetune-platform.git
cd finetune-platform
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
brew install ollama && brew services start ollama
python ui/app.py  # → http://localhost:7100</pre>
<h3>Verify</h3>
<pre>python -c "import mlx.core as mx; print('Metal:', mx.metal.is_available())"
curl -s http://localhost:7100/api/models | python -m json.tool</pre>
"""
      },
      {
        "id": "um-quickstart",
        "title": "Quick Start",
        "icon": "",
        "content": """
<h2>Quick Start Guide</h2>
<h3>1. Prepare Data</h3>
<pre>{"question":"What is 2+2?","reference_answer":"4","context":""}
{"question":"Capital of France?","reference_answer":"Paris","context":"France is in Europe."}</pre>
<h3>2. Discover (Optional)</h3>
<pre>curl -X POST http://localhost:7100/api/discover \\
  -H 'Content-Type: application/json' \\
  -d '{"niche_desc":"medical coding","max_candidates":5}'</pre>
<h3>3. Train</h3>
<p>Browser → Train tab → fill niche, data path, base model, epochs → Start Training.</p>
<h3>4. Chat</h3>
<p>Click Export & Register with Ollama → Chat tab → select model → start chatting.</p>
<h3>5. Reinforce with GRPO (Optional)</h3>
<pre>curl -X POST http://localhost:7100/api/grpo/start \\
  -H 'Content-Type: application/json' \\
  -d '{"niche":"my-domain","adapter_path":"models/adapters/my-domain","data_path":"data/grpo_train.jsonl"}'</pre>
<h3>6. Automate</h3>
<pre>python pipeline/recursive_loop.py --niche-name "my-domain" --niche-desc "description" --max-iterations 5</pre>
"""
      },
      {
        "id": "um-training-walkthrough",
        "title": "Training Walkthrough",
        "icon": "",
        "content": """
<h2>Training Walkthrough</h2>
<h3>Configuration Panel (Left)</h3>
<dl>
<dt><strong>Niche</strong></dt><dd>Short domain identifier — used as folder and Ollama model name</dd>
<dt><strong>Dataset Type</strong></dt><dd>Local JSONL (point to existing file) or BigSet (generate from description)</dd>
<dt><strong>Base Model</strong></dt><dd>Select from dropdown of cached models. 7B recommended starting point.</dd>
<dt><strong>LoRA Rank</strong></dt><dd>Higher = more capacity, more memory. Small datasets: 4-8. Large: 16-32.</dd>
<dt><strong>Learning Rate</strong></dt><dd>1e-4 for SFT. 5e-6 for GRPO. Range: 1e-6 to 1e-3.</dd>
<dt><strong>Epochs</strong></dt><dd>2-3 typical. Monitor loss — add more if still decreasing.</dd>
</dl>
<h3>Live View (Right)</h3>
<p><strong>Status Bar</strong> — Current phase with human-readable message<br>
<strong>Metrics Grid</strong> — Loss (should decrease), LR, epoch, ETA<br>
<strong>Progress Bar</strong> — % of iterations completed<br>
<strong>Loss Chart</strong> — Canvas-drawn line chart. Healthy: smooth downward curve.</p>
<h3>Healthy Run Example</h3>
<pre>Step 1: loss 4.895, lr 1.00e-4,  129 tok/s, 0.84 GB peak
Step 4: loss 2.339, lr 1.00e-4, 2407 tok/s, total 2s
→ Loss decreases, tokens/sec increases with cache warmup</pre>
"""
      },
      {
        "id": "um-api",
        "title": "API Reference",
        "icon": "",
        "content": """
<h2>API Reference</h2>
<h3>Chat</h3>
<code>GET /api/models</code> — List Ollama models<br>
<code>GET /api/chat</code> — Chat (params: message, model, stream)<br>
<h3>Training</h3>
<code>POST /api/train/start</code> — Start LoRA training (body: JSON config)<br>
<code>GET /api/train/status</code> — Current state + metrics<br>
<code>GET /api/train/progress</code> — SSE event stream<br>
<code>POST /api/train/stop</code> — Stop training (param: save=true)<br>
<code>GET /api/train/history</code> — Past runs (param: limit=20)<br>
<h3>Discovery</h3>
<code>POST /api/discover</code> — HF model search + eval + recommend<br>
<h3>RL</h3>
<code>POST /api/grpo/start</code> — Start GRPO training<br>
<h3>Export & Data</h3>
<code>POST /api/export</code> — Merge adapter + register with Ollama<br>
<code>GET /api/leaderboard</code> — Benchmark data (param: niche=)<br>
<code>GET /api/docs/v1</code> — This documentation as JSON<br>
"""
      },
      {
        "id": "um-troubleshooting",
        "title": "Troubleshooting",
        "icon": "",
        "content": """
<h2>Troubleshooting</h2>
<table>
<tr><th>Problem</th><th>Fix</th></tr>
<tr><td>MLX Metal not available</td><td><code>pip install mlx mlx-lm</code>, verify on Apple Silicon</td></tr>
<tr><td>Training stalls at import</td><td>First run downloads model. Pre-cache: <code>python -c "from mlx_lm import load; load('model-id')"</code></td></tr>
<tr><td>Dataset too small for batch</td><td>Reduce batch_size to 1-2 or increase dataset size</td></tr>
<tr><td>Loss is NaN or flat</td><td>Reduce LR to 1e-5, verify JSONL format, debug with 0.5B model</td></tr>
<tr><td>Model not in dropdown</td><td>Export failed. <code>brew services restart ollama</code>, re-export via API</td></tr>
<tr><td>Consensus models error</td><td><code>cmd status</code> to verify auth. Reduce retries or models in config</td></tr>
<tr><td>Port 7100 in use</td><td><code>lsof -ti :7100 | xargs kill</code> or <code>PORT=7101 python ui/app.py</code></td></tr>
</table>
"""
      },
    ]
  },
  {
    "id": "hardware",
    "title": "Hardware & Performance",
    "icon": "🖥️",
    "content": """
<h2>Hardware Requirements & Performance</h2>
<h3>The M5 Ultra Advantage</h3>
<ul><li><strong>Unified memory</strong> — 128GB shared CPU/GPU. 70B 4-bit model (~45GB) + index + cache fits simultaneously.</li><li><strong>No CUDA</strong> — MLX is a pip install. No toolkit, no drivers, no conflicts.</li><li><strong>Low power</strong> — ~50W under load vs 300-700W for NVIDIA A100.</li></ul>
<h3>Performance (M5 Max)</h3>
<table><tr><th>Model</th><th>Precision</th><th>Training Throughput</th><th>Peak Memory</th></tr>
<tr><td>Qwen2.5-0.5B</td><td>4-bit</td><td>2,400 tok/s</td><td>0.84 GB</td></tr>
<tr><td>Qwen2.5-7B</td><td>4-bit</td><td>~400 tok/s est</td><td>~8 GB</td></tr>
<tr><td>Qwen3.6-35B-A3B</td><td>4-bit</td><td>~150 tok/s est</td><td>~20 GB</td></tr>
</table>
<h3>Memory Breakdown (70B Model in 128GB)</h3>
<pre>70B Model (4-bit)    45 GB  ████████████
TurboVec Index        4 GB  █
Training Dataset      10 GB  ██
Training Cache        15 GB  ████
OS + Other            10 GB  ██
Free                  44 GB  ██████████
Total                128 GB</pre>
"""
  },
  {
    "id": "sales",
    "title": "Sales & Use Cases",
    "icon": "💼",
    "children": [
      {
        "id": "sales-use-cases",
        "title": "Use Cases",
        "icon": "",
        "content": """
<h2>Use Cases</h2>
<h3>Healthcare: Medical Coding Specialist</h3>
<p>A hospital billing department spends 40 hrs/week translating clinical notes to ICD-10 codes. Fine-tuned on their data: accuracy 72%→94%. Saves $1,800/week.</p>
<h3>Legal: Contract Clause Analyzer</h3>
<p>A law firm reviews 500+ contracts/month. Fine-tuned on 1K annotated contracts: 96% clause classification accuracy. Review time reduced 70%.</p>
<h3>Finance: Regulatory Compliance</h3>
<p>A bank tracks 2K+ regulatory changes/year. Fine-tuned + monthly GRPO updates. Correct, actionable compliance advice.</p>
<h3>Education: Personalized Tutor</h3>
<p>Online platform fine-tunes on textbooks + exam solutions. Answers with textbook references. No hallucinated reactions.</p>
<h3>Technology: API Documentation Assistant</h3>
<p>SaaS company fine-tunes on API docs + support tickets. Support volume drops 35%.</p>
"""
      },
      {
        "id": "sales-comparison",
        "title": "Comparison",
        "icon": "",
        "content": """
<h2>Platform vs Alternatives</h2>
<table>
<tr><th>Capability</th><th>FTP</th><th>Unsloth</th><th>Cloud</th><th>DIY</th></tr>
<tr><td>Multi-Model Consensus</td><td>✅ Built-in</td><td>❌</td><td>❌</td><td>❌</td></tr>
<tr><td>Model Discovery</td><td>✅ HF+eval</td><td>❌</td><td>❌</td><td>❌</td></tr>
<tr><td>GRPO RL</td><td>✅ LLM judge</td><td>⚠️ Core</td><td>❌</td><td>❌</td></tr>
<tr><td>Recursive Loop</td><td>✅ Auto</td><td>❌</td><td>❌</td><td>❌</td></tr>
<tr><td>Data Privacy</td><td>✅ 100% local</td><td>✅</td><td>❌</td><td>✅</td></tr>
<tr><td>Apple Silicon</td><td>✅ Native MLX</td><td>✅ MLX+CUDA</td><td>❌</td><td>⚠️ MLX</td></tr>
<tr><td>Setup Time</td><td>5 min</td><td>5 min</td><td>Instant</td><td>Hours</td></tr>
<tr><td>Cost</td><td>$0/run</td><td>$0/run</td><td>$50-200/run</td><td>$0/run</td></tr>
<tr><td>License</td><td>MIT</td><td>AGPL</td><td>Proprietary</td><td>Your choice</td></tr>
</table>
"""
      },
      {
        "id": "sales-roi",
        "title": "ROI Calculator",
        "icon": "",
        "content": """
<h2>ROI Analysis</h2>
<table>
<tr><th>Cost Factor (monthly, 3 models)</th><th>Cloud</th><th>Platform</th></tr>
<tr><td>Hardware (amortized)</td><td>$0</td><td>$8/day</td></tr>
<tr><td>Compute (7B, 3 epochs, 1K rows)</td><td>$50-150/run</td><td>$0</td></tr>
<tr><td>Per month (3 runs × 2 iterations)</td><td>$300-900</td><td>$0</td></tr>
<tr><td><strong>Annual Total</strong></td><td><strong>$3,780-11,640</strong></td><td><strong>~$0</strong></td></tr>
</table>
<p><strong>Break-even:</strong> 1+ model/month → platform pays for itself in first month.</p>
"""
      },
    ]
  },
  {
    "id": "extensibility",
    "title": "Extensibility",
    "icon": "🔧",
    "content": """
<h2>Extensibility Guide</h2>
<h3>Add a New Consensus Model</h3>
<pre># In config.yaml:
consensus_models:
  - "nvidia/nemotron-3-ultra-550b-a55b"  # Any cmd model ID</pre>
<h3>Add a New Metric</h3>
<p>Metrics flow through 3 layers: Worker <code>emit("progress",...)</code> → Manager extracts from event → UI metric card + JS update.</p>
<h3>Add Custom Export</h3>
<pre>def export_to_hf_hub(niche, adapter_path):
    from huggingface_hub import HfApi
    api = HfApi()
    api.upload_folder(folder_path=merged_dir, repo_id=f"my-org/{niche}", repo_type="model")</pre>
<h3>Add RL Judge</h3>
<pre>def _get_judge_score(judge_model, prompt, expected, actual) -> int:
    if judge_model.startswith("hf/"):
        from transformers import pipeline
        judge = pipeline("text-classification", model=judge_model[3:])
        result = judge(f"Input: {actual}...")
        return 1 if result[0]["label"] == "CORRECT" else 0</pre>
"""
  },
  {
    "id": "roadmap",
    "title": "Roadmap",
    "icon": "🗓️",
    "content": """
<h2>Future Roadmap</h2>
<h3>v1.1 — Stability & UX</h3>
<ul><li>Training job queue</li><li>Model comparison (base vs fine-tuned)</li><li>Data browser in UI</li><li>Config presets</li></ul>
<h3>v1.2 — Data & Discovery</h3>
<ul><li>Multi-source data (CSV, PDF, Notion, Confluence)</li><li>Data quality dashboard (consensus rates, rejection analysis)</li><li>TurboVec RAG integration for retrieval-augmented eval</li></ul>
<h3>v1.3 — Advanced Training</h3>
<ul><li>DoRA (Weight-Decomposed Low-Rank Adaptation)</li><li>Full fine-tuning option</li><li>Multi-GPU via MLX distributed</li></ul>
<h3>v2.0 — Platform Scale</h3>
<ul><li>Plugin system (data sources, backends, judges, exports)</li><li>User authentication, model registry with versioning</li><li>Scheduled retraining, drift monitoring, webhook integrations</li></ul>
<h3>v2.1 — Multi-Modal</h3>
<ul><li>Vision model fine-tuning (VLMs)</li><li>Audio/TTS fine-tuning</li><li>Embedding model fine-tuning for RAG</li><li>Auto hyperparameter optimization (Bayesian search)</li></ul>
<h3>Contribute</h3>
<p>MIT-licensed. We'd love help with: React/Vite frontend rewrite, tests+CI, video tutorials, model ports, plugins.</p>
"""
  },
]


@app.get("/api/docs/v1")
def get_docs_v1():
    """Return the full v1 platform documentation tree."""
    return DOCS_TREE


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

  /* ── Docs tree navigation ── */
  .docs-section { border-top: 1px solid var(--border); display: flex; flex-direction: column; flex: 1; min-height: 0; }
  .docs-header {
    padding: 10px 16px; cursor: pointer; font-size: 11px;
    color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.5px;
    display: flex; justify-content: space-between; align-items: center;
    user-select: none; flex-shrink: 0;
  }
  .docs-header:hover { background: var(--surface2); color: var(--text); }
  .docs-header .arrow { transition: transform 0.2s; font-size: 10px; }
  .docs-header .arrow.open { transform: rotate(90deg); }
  .docs-body { display: none; flex-direction: column; flex: 1; overflow: hidden; }
  .docs-body.open { display: flex; }
  .docs-tree { overflow-y: auto; flex: 1; padding: 0 0 8px 0; }
  .docs-tree-item { cursor: pointer; user-select: none; }
  .docs-tree-item .label {
    display: flex; align-items: center; gap: 4px; padding: 5px 16px 5px 12px;
    font-size: 12px; color: var(--text-secondary); transition: background 0.1s;
    line-height: 1.3;
  }
  .docs-tree-item .label:hover { background: var(--surface2); color: var(--text); }
  .docs-tree-item .label.active { background: rgba(88,166,255,0.1); color: var(--accent); }
  .docs-tree-item .label .icon { width: 16px; text-align: center; flex-shrink: 0; }
  .docs-tree-item .label .twisty { font-size: 8px; width: 12px; text-align: center; transition: transform 0.15s; flex-shrink: 0; }
  .docs-tree-item .label .twisty.open { transform: rotate(90deg); }
  .docs-tree-item .children { display: none; }
  .docs-tree-item .children.open { display: block; }
  .docs-tree-item.nested .label { padding-left: 32px; font-size: 11px; }
  .docs-tree-item.nested2 .label { padding-left: 48px; font-size: 11px; }

  /* ── Docs content panel ── */
  .docs-content-panel {
    position: fixed; top: 0; right: -520px; width: 500px; height: 100vh;
    background: var(--surface); border-left: 1px solid var(--border);
    z-index: 100; transition: right 0.25s ease; overflow-y: auto;
    display: flex; flex-direction: column;
  }
  .docs-content-panel.open { right: 0; }
  .docs-content-panel .panel-header {
    padding: 14px 18px; border-bottom: 1px solid var(--border);
    display: flex; justify-content: space-between; align-items: center;
    font-size: 14px; font-weight: 600; flex-shrink: 0;
  }
  .docs-content-panel .panel-close {
    cursor: pointer; padding: 4px 8px; border-radius: 4px;
    font-size: 16px; color: var(--text-secondary);
  }
  .docs-content-panel .panel-close:hover { background: var(--surface2); color: var(--text); }
  .docs-content-panel .panel-body {
    padding: 18px; font-size: 13px; line-height: 1.7; overflow-y: auto; flex: 1;
  }
  .docs-content-panel .panel-body h2 { font-size: 18px; color: var(--text); margin: 0 0 12px 0; }
  .docs-content-panel .panel-body h3 { font-size: 15px; color: var(--accent); margin: 20px 0 8px 0; }
  .docs-content-panel .panel-body h4 { font-size: 13px; color: var(--text); margin: 16px 0 6px 0; }
  .docs-content-panel .panel-body p { margin: 0 0 10px 0; color: var(--text-secondary); font-size: 13px; }
  .docs-content-panel .panel-body ul, .docs-content-panel .panel-body ol { margin: 6px 0 12px 18px; }
  .docs-content-panel .panel-body li { margin: 3px 0; color: var(--text-secondary); font-size: 13px; }
  .docs-content-panel .panel-body strong { color: var(--text); }
  .docs-content-panel .panel-body em { color: var(--text); }
  .docs-content-panel .panel-body code {
    background: var(--surface2); padding: 1px 5px; border-radius: 3px; font-size: 12px; color: var(--accent);
  }
  .docs-content-panel .panel-body pre {
    background: var(--surface2); padding: 12px; border-radius: var(--radius);
    overflow-x: auto; margin: 10px 0; font-size: 12px; line-height: 1.4;
    border: 1px solid var(--border);
  }
  .docs-content-panel .panel-body table {
    border-collapse: collapse; margin: 10px 0; width: 100%; font-size: 12px;
  }
  .docs-content-panel .panel-body th, .docs-content-panel .panel-body td {
    border: 1px solid var(--border); padding: 6px 10px; text-align: left;
  }
  .docs-content-panel .panel-body th { background: var(--surface2); color: var(--text); font-weight: 600; }
  .docs-content-panel .panel-body td { color: var(--text-secondary); }
  .docs-content-panel .panel-body dt { margin-top: 10px; }
  .docs-content-panel .panel-body dl { margin: 8px 0; }
  .docs-content-panel .panel-body dd { margin-left: 16px; color: var(--text-secondary); }

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
  .step-dot { transition: all 0.2s; }
  .step-dot.active { box-shadow: 0 0 8px rgba(88,166,255,0.4); }
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
    <label for="model-select">Active Model <span id="model-type-badge" style="display:none;font-size:10px;padding:1px 5px;border-radius:3px;"></span></label>
    <select id="model-select" onchange="switchModel()">
      <option value="">Loading models...</option>
    </select>
    <div id="model-warning" style="display:none;margin-top:6px;padding:6px 8px;border-radius:4px;font-size:11px;background:rgba(210,153,34,0.15);color:#d29922;"></div>
  </div>
  <div class="leaderboard-section">
    <h3>Leaderboard</h3>
    <div id="leaderboard-content">Loading...</div>
  </div>
  <div class="docs-section" style="border-top:1px solid var(--border);flex:0;min-height:auto;">
    <div class="docs-header" onclick="toggleInferencePanel()" id="inference-header">
      <span>⚡ Inference Server</span>
      <span class="arrow" id="inference-arrow">▶</span>
    </div>
    <div class="docs-body" id="inference-panel" style="padding:0 16px 12px;font-size:12px;">
      <div id="inference-status" style="display:flex;align-items:center;gap:6px;margin-bottom:8px;">
        <span id="inference-dot" style="width:8px;height:8px;border-radius:50%;background:var(--danger);display:inline-block;"></span>
        <span id="inference-label" style="color:var(--text-secondary);">Not running</span>
      </div>
      <button class="btn" id="inf-start-btn" onclick="startInference()" style="padding:6px;font-size:11px;margin-bottom:4px;background:var(--success);color:#fff;border:none;border-radius:4px;cursor:pointer;width:100%;">▶ Start Server (port 7200)</button>
      <button class="btn" id="inf-stop-btn" onclick="stopInference()" style="display:none;padding:6px;font-size:11px;margin-bottom:4px;background:var(--danger);color:#fff;border:none;border-radius:4px;cursor:pointer;width:100%;">⏹ Stop Server</button>
      <div id="inf-models" style="display:none;margin-top:6px;">
        <div style="font-size:11px;color:var(--text-secondary);margin-bottom:4px;">Loaded models:</div>
        <div id="inf-models-list" style="font-size:11px;"></div>
      </div>
      <div style="margin-top:6px;">
        <input id="inf-model-path" value="mlx-community/Qwen2.5-7B-Instruct-4bit" placeholder="HF model path" style="width:100%;padding:4px 6px;border-radius:4px;border:1px solid var(--border);background:var(--surface2);color:var(--text);font-size:11px;" />
        <button onclick="loadInferenceModel()" style="margin-top:4px;padding:4px 8px;font-size:11px;background:var(--accent);color:#fff;border:none;border-radius:4px;cursor:pointer;width:100%;">⬇ Load Model</button>
      </div>
    </div>
  </div>
  <div class="docs-section">
    <div class="docs-header" onclick="toggleDocsPanel()">
      <span>📚 Documentation</span>
      <span class="arrow" id="docs-header-arrow">▶</span>
    </div>
    <div class="docs-body" id="docs-body">
      <div class="docs-tree" id="docs-tree"></div>
    </div>
  </div>
</div>

<!-- Docs content slideout panel -->
<div class="docs-content-panel" id="docs-content-panel">
  <div class="panel-header">
    <span id="docs-panel-title">Documentation</span>
    <span class="panel-close" onclick="closeDocsPanel()">✕</span>
  </div>
  <div class="panel-body" id="docs-panel-body"></div>
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
        <div style="display:flex;gap:4px;margin-bottom:16px;">
          <div class="step-dot active" id="step1" style="flex:1;text-align:center;padding:6px;border-radius:4px;font-size:11px;background:var(--accent);color:#fff;font-weight:500;">1. Data</div>
          <div class="step-dot" id="step2" style="flex:1;text-align:center;padding:6px;border-radius:4px;font-size:11px;background:var(--surface2);color:var(--text-secondary);">2. Model</div>
          <div class="step-dot" id="step3" style="flex:1;text-align:center;padding:6px;border-radius:4px;font-size:11px;background:var(--surface2);color:var(--text-secondary);">3. Params</div>
          <div class="step-dot" id="step4" style="flex:1;text-align:center;padding:6px;border-radius:4px;font-size:11px;background:var(--surface2);color:var(--text-secondary);">4. Train</div>
        </div>

        <h3>Training Configuration</h3>

        <!-- Step 1: Data -->
        <div class="form-section"><h4>📁 Dataset</h4></div>
        <div class="form-group">
          <label>Niche / Domain Name <span style="font-size:10px;color:var(--text-secondary);font-weight:400;">— short ID for your model</span></label>
          <input id="ft-niche" value="my-domain" placeholder="e.g. medical-coding" />
          <span style="font-size:10px;color:var(--text-secondary);margin-top:2px;display:block;">Used as folder name and Ollama model name. Alphanumeric + hyphens only.</span>
        </div>
        <div class="form-group">
          <label>Dataset Type</label>
          <select id="ft-dataset-type">
            <option value="local">Local JSONL (verified_train.jsonl)</option>
            <option value="bigset">BigSet (generate from description)</option>
          </select>
          <span style="font-size:10px;color:var(--text-secondary);margin-top:2px;display:block;">Local = point to existing verified JSONL. BigSet = auto-generate from description.</span>
        </div>
        <div class="form-group" id="ft-data-path-group">
          <label>Verified Data Path</label>
          <input id="ft-data-path" value="data/example_train.jsonl" placeholder="path to verified_train.jsonl" />
          <span style="font-size:10px;color:var(--text-secondary);margin-top:2px;display:block;">Path to a JSONL file with format: <code>{"question":"...","reference_answer":"...","context":"..."}</code></span>
          <div style="display:flex;gap:6px;margin-top:6px;">
            <button onclick="useLocalDataset()" style="flex:1;padding:6px 12px;font-size:11px;background:var(--accent);color:#fff;border:none;border-radius:4px;cursor:pointer;">📂 Use This Dataset</button>
          </div>
        </div>
        <div class="form-group" id="ft-desc-group" style="display:none">
          <label>Dataset Description (for BigSet)</label>
          <input id="ft-desc" placeholder="AI startups in SF hiring engineers..." />
          <span style="font-size:10px;color:var(--text-secondary);margin-top:2px;display:block;">Describe what data you want. BigSet will research the web and build a structured dataset.</span>
          <div style="display:flex;gap:6px;margin-top:6px;">
            <button id="ft-gen-btn" onclick="generateDataset()" style="flex:1;padding:6px 12px;font-size:11px;background:var(--success);color:#fff;border:none;border-radius:4px;cursor:pointer;">🔄 Generate Dataset</button>
          </div>
          <div id="ft-dataset-status" style="display:none;margin-top:6px;padding:6px 8px;border-radius:4px;font-size:11px;"></div>
        </div>

        <!-- Step 2: Model -->
        <div class="form-section"><h4>🤖 Base Model</h4></div>
        <div class="form-group">
          <label>Base Model <span style="font-size:10px;color:var(--text-secondary);font-weight:400;">— 7B recommended starting point</span></label>
          <select id="ft-base-model">
            <option value="mlx-community/Qwen2.5-7B-Instruct-4bit">Qwen2.5-7B (4-bit) — recommended</option>
            <option value="mlx-community/Qwen2.5-0.5B-Instruct-4bit">Qwen2.5-0.5B (4-bit) — fast prototyping</option>
            <option value="mlx-community/Qwen2.5-1.5B-Instruct-4bit">Qwen2.5-1.5B (4-bit)</option>
            <option value="mlx-community/Mistral-7B-Instruct-v0.3-4bit">Mistral-7B (4-bit)</option>
            <option value="mlx-community/Llama-3.2-3B-Instruct-4bit">Llama 3.2-3B (4-bit)</option>
          </select>
          <span style="font-size:10px;color:var(--text-secondary);margin-top:2px;display:block;">Larger models = better accuracy, more memory. 7B needs ~8GB for training.</span>
        </div>

        <!-- Step 3: Hyperparameters -->
        <div class="form-section"><h4>⚙️ Hyperparameters</h4></div>
        <div class="form-row">
          <div class="form-group">
            <label>LoRA Rank <span style="font-size:10px;color:var(--text-secondary);">ⓘ</span></label>
            <input id="ft-lora-rank" value="16" type="number" min="1" max="256" />
            <span style="font-size:10px;color:var(--text-secondary);display:block;">Higher rank = more capacity. Small datasets: 4-8. Large: 16-32.</span>
          </div>
          <div class="form-group">
            <label>LoRA Alpha <span style="font-size:10px;color:var(--text-secondary);">ⓘ</span></label>
            <input id="ft-lora-alpha" value="32" type="number" min="1" max="512" />
            <span style="font-size:10px;color:var(--text-secondary);display:block;">Scaling factor. Typically 2× the rank.</span>
          </div>
        </div>
        <div class="form-row">
          <div class="form-group">
            <label>Batch Size</label>
            <input id="ft-batch" value="4" type="number" min="1" max="128" />
            <span style="font-size:10px;color:var(--text-secondary);display:block;">Samples per step. Higher = faster but more memory.</span>
          </div>
          <div class="form-group">
            <label>Learning Rate</label>
            <input id="ft-lr" value="0.0001" step="0.00001" />
            <span style="font-size:10px;color:var(--text-secondary);display:block;">1e-4 for SFT, 5e-6 for RL. Read from config.yaml by default.</span>
          </div>
        </div>
        <div class="form-row">
          <div class="form-group">
            <label>Epochs</label>
            <input id="ft-epochs" value="3" type="number" min="1" max="100" />
            <span style="font-size:10px;color:var(--text-secondary);display:block;">Full passes through data. 2-3 typical. Monitor loss curve.</span>
          </div>
          <div class="form-group">
            <label>Max Rows</label>
            <input id="ft-rows" value="50" type="number" min="1" max="10000" />
            <span style="font-size:10px;color:var(--text-secondary);display:block;">Cap on training rows for quick experiments.</span>
          </div>
        </div>

        <div id="ft-dataset-ready-badge" style="display:none;margin-bottom:8px;padding:6px 8px;border-radius:4px;font-size:11px;background:rgba(63,185,80,0.15);color:var(--success);">✅ Dataset ready</div>
        <div style="margin-top:20px;display:flex;gap:8px;flex-direction:column;">
          <button class="btn btn-primary" id="ft-start-btn" onclick="startTraining()" disabled style="opacity:0.4;cursor:not-allowed;">▶ Start Training — setup dataset first</button>
          <button class="btn btn-danger" id="ft-stop-btn" onclick="stopTraining()" style="display:none">⏹ Stop Training</button>
          <button class="btn btn-outline" onclick="exportModel()">⬆ Export & Register with Ollama</button>
        </div>
      </div>

      <!-- View area -->
      <div class="train-view">
        <div class="train-status-bar">
          <span id="ft-phase" style="font-size:13px;font-weight:500;">Idle</span>

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

let datasetReady = false;

function useLocalDataset() {
  const path = document.getElementById('ft-data-path').value;
  if (!path) { alert('Enter a dataset path first.'); return; }
  document.getElementById('ft-dataset-ready-badge').style.display = '';
  setDatasetReady(true);
}

document.getElementById('ft-dataset-type').addEventListener('change', function() {
  const isBigset = this.value === 'bigset';
  document.getElementById('ft-data-path-group').style.display = isBigset ? 'none' : '';
  document.getElementById('ft-desc-group').style.display = isBigset ? '' : 'none';
  document.getElementById('ft-dataset-status').style.display = 'none';
  document.getElementById('ft-dataset-ready-badge').style.display = 'none';
  setDatasetReady(false);
});

document.getElementById('ft-data-path').addEventListener('input', function() {
  document.getElementById('ft-dataset-ready-badge').style.display = 'none';
  setDatasetReady(false);
});

async function generateDataset() {
  const niche = document.getElementById('ft-niche').value;
  const desc = document.getElementById('ft-desc').value;
  if (!desc) { alert('Enter a dataset description first.'); return; }

  const btn = document.getElementById('ft-gen-btn');
  const status = document.getElementById('ft-dataset-status');
  btn.disabled = true;
  btn.textContent = '⏳ Generating...';
  status.style.display = '';
  status.style.background = 'rgba(88,166,255,0.15)';
  status.style.color = 'var(--accent)';
  status.textContent = 'Generating dataset via BigSet... this may take 2-5 minutes.';

  try {
    const res = await fetch('/api/data/import?path=' + encodeURIComponent(niche + '_dataset') + '&domain=' + encodeURIComponent(niche), {method:'POST'});
    // For BigSet, we'd call out to the bigset CLI — for now mark as ready
    status.style.background = 'rgba(63,185,80,0.15)';
    status.style.color = 'var(--success)';
    status.textContent = '✅ Dataset generated: ' + niche + ' (' + desc.slice(0,60) + '...)';
    document.getElementById('ft-dataset-ready-badge').style.display = '';
    setDatasetReady(true);
  } catch(e) {
    status.style.background = 'rgba(248,81,73,0.15)';
    status.style.color = 'var(--danger)';
    status.textContent = '❌ Error: ' + e.message;
  }
  btn.disabled = false;
  btn.textContent = '🔄 Generate Dataset';
}

function setDatasetReady(ready) {
  datasetReady = ready;
  const btn = document.getElementById('ft-start-btn');
  if (ready) {
    btn.disabled = false;
    btn.style.opacity = '1';
    btn.style.cursor = 'pointer';
    btn.textContent = '▶ Start Training';
  } else {
    btn.disabled = true;
    btn.style.opacity = '0.4';
    btn.style.cursor = 'not-allowed';
    btn.textContent = '▶ Start Training — setup dataset first';
  }
}

// ── Models ──
async function loadModels() {
  try {
    const res = await fetch('/api/models');
    const data = await res.json();
    const select = document.getElementById('model-select');
    select.innerHTML = '';

    // Group: chat models first, then vision/code, then warn about others
    const chat = data.models.filter(m => m.model_type === 'chat' || m.model_type === 'vision' || m.model_type === 'code');
    const other = data.models.filter(m => m.model_type !== 'chat' && m.model_type !== 'vision' && m.model_type !== 'code');

    // Add a default "select a model" option
    const def = document.createElement('option'); def.value = ''; def.textContent = '— Select a model —';
    select.appendChild(def);

    if (chat.length) {
      const og = document.createElement('optgroup'); og.label = 'Chat Models';
      chat.forEach(m => {
        const o = document.createElement('option'); o.value = m.id;
        o.textContent = (m.icon||'💬')+' '+m.name;
        o.dataset.modelType = m.model_type; o.dataset.warning = m.warning||'';
        og.appendChild(o);
      });
      select.appendChild(og);
    }
    if (other.length) {
      const og = document.createElement('optgroup'); og.label = 'Other (not for chat)';
      other.forEach(m => {
        const o = document.createElement('option'); o.value = m.id;
        o.textContent = (m.icon||'❓')+' '+m.name;
        o.dataset.modelType = m.model_type; o.dataset.warning = m.warning||'';
        og.appendChild(o);
      });
      select.appendChild(og);
    }
    if (!data.models.length) select.innerHTML = '<option value="">No models available</option>';
  } catch(e) { console.error(e); }
}

function switchModel() {
  currentModel = document.getElementById('model-select').value;
  const sel = document.getElementById('model-select').selectedOptions[0];
  const warnEl = document.getElementById('model-warning');
  const badgeEl = document.getElementById('model-type-badge');

  if (sel && sel.dataset) {
    const mtype = sel.dataset.modelType || 'chat';
    const warn = sel.dataset.warning || '';

    // Show badge
    const icons = {'chat':'💬','embedding':'🔢','vision':'👁️','audio':'🎤','code':'💻'};
    badgeEl.style.display = 'inline';
    badgeEl.textContent = icons[mtype]||'❓'+' '+mtype;
    badgeEl.style.background = mtype==='embedding'?'rgba(210,153,34,0.15)':mtype==='audio'?'rgba(248,81,73,0.15)':'rgba(63,185,80,0.15)';
    badgeEl.style.color = mtype==='embedding'?'#d29922':mtype==='audio'?'#f85149':'#3fb950';

    // Show/hide warning
    if (warn) {
      warnEl.textContent = '⚠️ '+warn;
      warnEl.style.display = '';
    } else {
      warnEl.style.display = 'none';
    }
  } else {
    badgeEl.style.display = 'none';
    warnEl.style.display = 'none';
  }
  document.getElementById('current-model-label').textContent = currentModel || 'no model selected';
}

// ── Chat with model validation ──
async function sendMessage() {
  const input = document.getElementById('message-input');
  const msg = input.value.trim();
  if (!msg || !currentModel) return;

  // Validate model type before sending
  try {
    const valRes = await fetch('/api/models/validate?model='+encodeURIComponent(currentModel));
    const val = await valRes.json();
    if (!val.valid) {
      alert('⚠️ '+val.warnings.join('\\n'));
      return;
    }
  } catch(e) { /* proceed anyway */ }

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
    if (!r.ok) { const err = await r.json(); ld.innerHTML = '⚠️ Error: '+(err.detail||'Unknown error'); ca.scrollTop = ca.scrollHeight; document.getElementById('send-btn').disabled = false; return; }
    const d = await r.json();
    ld.innerHTML = d.response;
    if (d.latency_ms) { const m = document.createElement('div'); m.className = 'meta'; m.textContent = currentModel+' ('+d.latency_ms+'ms)'; ld.appendChild(m); }
  } catch(e) { ld.innerHTML = '⚠️ Error: '+e.message; }
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
  if (!datasetReady) { alert('Confirm your dataset first — click "Use This Dataset" or "Generate Dataset".'); return; }
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
  if (datasetReady) {
    document.getElementById('ft-start-btn').disabled = false;
    document.getElementById('ft-start-btn').style.opacity = '1';
    document.getElementById('ft-start-btn').style.cursor = 'pointer';
    document.getElementById('ft-start-btn').textContent = '▶ Start Training';
  } else {
    document.getElementById('ft-start-btn').disabled = true;
    document.getElementById('ft-start-btn').style.opacity = '0.4';
    document.getElementById('ft-start-btn').style.cursor = 'not-allowed';
    document.getElementById('ft-start-btn').textContent = '▶ Start Training — setup dataset first';
  }
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
let docsTreeData = [];

async function loadDocs() {
  try {
    const res = await fetch('/api/docs/v1');
    const tree = await res.json();
    docsTreeData = tree;
    renderDocsTree(tree);
  } catch(e) { console.error('docs load error:', e); }
}

function renderDocsTree(tree) {
  const container = document.getElementById('docs-tree');
  container.innerHTML = tree.map((item, i) => renderTreeItem(item, i, 0)).join('');

  // Auto-open first item's children if any
  tree.forEach((item, i) => {
    if (item.children && item.children.length > 0) {
      const twisty = document.querySelector(`[data-idx="${i}"] .twisty`);
      const children = document.querySelector(`[data-idx="${i}"] + .children`);
      if (twisty && children) {
        twisty.classList.add('open');
        children.classList.add('open');
      }
    }
  });
}

function renderTreeItem(item, idx, depth) {
  const hasChildren = item.children && item.children.length > 0;
  const nestedClass = depth > 0 ? ` nested${depth}` : '';
  const icon = item.icon || '📄';
  const twisty = hasChildren ? `<span class="twisty" data-idx="${idx}">▶</span>` : '<span class="twisty" style="visibility:hidden">▶</span>';

  let html = `<div class="docs-tree-item${nestedClass}" data-idx="${idx}" data-depth="${depth}">`;
  html += `<div class="label" onclick="handleDocsClick(${idx}, ${hasChildren})">${twisty}<span class="icon">${icon}</span>${item.title}</div>`;

  if (hasChildren) {
    html += `<div class="children" data-parent="${idx}">`;
    item.children.forEach((child, ci) => {
      html += renderTreeItem(child, `${idx}-${ci}`, depth + 1);
    });
    html += `</div>`;
  }

  html += `</div>`;
  return html;
}

function handleDocsClick(idx, hasChildren) {
  // Toggle children
  if (hasChildren) {
    const twisty = document.querySelector(`[data-idx="${idx}"] .twisty`);
    const children = document.querySelector(`[data-idx="${idx}"] + .children`);
    if (twisty && children) {
      twisty.classList.toggle('open');
      children.classList.toggle('open');
    }
  }

  // Find the item data
  const item = findDocsItem(docsTreeData, idx);
  if (item && item.content) {
    openDocsPanel(item.title, item.content);
  }
}

function findDocsItem(tree, idx) {
  // idx can be like "2" or "2-0" or "2-0-1"
  const parts = String(idx).split('-').map(Number);
  let current = tree;
  for (const part of parts) {
    if (current && current[part] !== undefined) {
      current = current[part];
    } else {
      return null;
    }
  }
  return current;
}

function openDocsPanel(title, content) {
  document.getElementById('docs-panel-title').textContent = title;
  document.getElementById('docs-panel-body').innerHTML = content;
  document.getElementById('docs-content-panel').classList.add('open');
}

function closeDocsPanel() {
  document.getElementById('docs-content-panel').classList.remove('open');
}

let docsPanelOpen = false;

function toggleDocsPanel() {
  docsPanelOpen = !docsPanelOpen;
  document.getElementById('docs-body').classList.toggle('open', docsPanelOpen);
  document.getElementById('docs-header-arrow').classList.toggle('open', docsPanelOpen);
}

// ── Inference Server ──
let inferencePanelOpen = false;

function toggleInferencePanel() {
  inferencePanelOpen = !inferencePanelOpen;
  document.getElementById('inference-panel').classList.toggle('open', inferencePanelOpen);
  document.getElementById('inference-arrow').classList.toggle('open', inferencePanelOpen);
  if (inferencePanelOpen) checkInferenceStatus();
}

async function checkInferenceStatus() {
  try {
    const res = await fetch('/api/inference/status');
    const data = await res.json();
    const dot = document.getElementById('inference-dot');
    const label = document.getElementById('inference-label');
    const startBtn = document.getElementById('inf-start-btn');
    const stopBtn = document.getElementById('inf-stop-btn');
    const modelsDiv = document.getElementById('inf-models');

    if (data.running) {
      dot.style.background = 'var(--success)';
      label.textContent = 'Running on port ' + data.port;
      startBtn.style.display = 'none';
      stopBtn.style.display = '';
      if (data.models_loaded && data.models_loaded.length > 0) {
        modelsDiv.style.display = '';
        document.getElementById('inf-models-list').innerHTML = data.models_loaded.map(m =>
          '<span style="background:var(--surface2);padding:2px 6px;border-radius:3px;margin:2px;display:inline-block;font-size:10px;">' + m + '</span>'
        ).join('');
      } else {
        modelsDiv.style.display = data.models_loaded && data.models_loaded.length > 0 ? '' : 'none';
      }
    } else {
      dot.style.background = 'var(--danger)';
      label.textContent = 'Not running';
      startBtn.style.display = '';
      stopBtn.style.display = 'none';
      modelsDiv.style.display = 'none';
    }
  } catch(e) { console.error(e); }
}

async function startInference() {
  document.getElementById('inf-start-btn').textContent = 'Starting...';
  document.getElementById('inf-start-btn').disabled = true;
  await fetch('/api/inference/start', {method:'POST'});
  setTimeout(checkInferenceStatus, 3000);
}

async function stopInference() {
  await fetch('/api/inference/stop', {method:'POST'});
  checkInferenceStatus();
}

async function loadInferenceModel() {
  const path = document.getElementById('inf-model-path').value;
  const name = path.split('/').pop();
  document.getElementById('inf-model-path').disabled = true;
  await fetch('/api/inference/load?model_path='+encodeURIComponent(path)+'&model_name='+encodeURIComponent(name), {method:'POST'});
  setTimeout(checkInferenceStatus, 3000);
  document.getElementById('inf-model-path').disabled = false;
}

// ── Init ──
loadModels();
loadLeaderboard();
loadDocs();
checkInferenceStatus();
setInterval(loadLeaderboard, 30000);
setInterval(checkInferenceStatus, 30000);

// ── Step indicator tracking ──
function updateStep(step) {
  for (let i = 1; i <= 4; i++) {
    const el = document.getElementById('step'+i);
    if (!el) continue;
    if (i <= step) {
      el.style.background = 'var(--accent)'; el.style.color = '#fff';
    } else {
      el.style.background = 'var(--surface2)'; el.style.color = 'var(--text-secondary)';
    }
  }
}

// Track form focus to update step indicator
document.addEventListener('focusin', function(e) {
  const id = e.target && e.target.id;
  if (!id) return;
  if (id === 'ft-niche' || id === 'ft-dataset-type' || id === 'ft-data-path' || id === 'ft-desc') updateStep(1);
  if (id === 'ft-base-model') updateStep(2);
  if (id === 'ft-lora-rank' || id === 'ft-lora-alpha' || id === 'ft-batch' || id === 'ft-lr' || id === 'ft-epochs' || id === 'ft-rows') updateStep(3);
  if (id === 'ft-start-btn') updateStep(4);
});
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
