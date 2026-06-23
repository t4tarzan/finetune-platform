# Fine-Tuning Platform

**Local recursive fine-tuning platform for Apple Silicon (M-series).**  
Generate datasets → verify via multi-model consensus → fine-tune with MLX LoRA → benchmark → serve via Ollama — all in one unified web UI.

![Platform UI](https://img.shields.io/badge/UI-FastAPI%20%2B%20HTML%2FCSS-blue)
![ML Framework](https://img.shields.io/badge/ML-MLX%20(Apple%20Native)-green)
![Python](https://img.shields.io/badge/Python-3.13-blue)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Platform](https://img.shields.io/badge/Platform-Apple%20Silicon%20(M%20Series)-orange)

---

## Features

| Feature | Description |
|---------|-------------|
| **Multi-Model Consensus** | 3+ commandcode API models (DeepSeek, Qwen, Kimi, Claude) verify each training row with domain-weighted adjudication; only consensus-passing data trains |
| **MLX LoRA Fine-Tuning** | Apple-native ML framework, subprocess-isolated training with live SSE progress streaming |
| **GRPO Reinforcement Learning** | Group Relative Policy Optimization with binary LLM judge reward (1=correct, 0=incorrect) |
| **Model Discovery Agent** | Searches HuggingFace for pre-trained models matching your niche, evaluates locally, recommends use-as-is vs fine-tune |
| **DuckDB Data Layer** | Columnar storage via DuckDB, Parquet/CSV/JSONL import, SQL-based training set generation, customer schema registration |
| **Persistent Inference API** | OpenAI-compatible endpoint on port 7200, models stay hot in memory, API key auth |
| **Recursive Improvement Loop** | Automate: generate data → discover → verify → train → eval → find gaps → repeat |
| **Live Training UI** | Step-by-step wizard, real-time loss chart, progress bar, metric grid (loss/lr/epoch/ETA/tokens/sec) |
| **Ollama Integration** | Fine-tuned models auto-register in Ollama → appear in dropdown immediately |
| **Benchmark Leaderboard** | Tracks accuracy/grounding/consistency per iteration with baseline deltas |
| **Compliance Adjudication** | Domain-weighted consensus for GDPR/HIPAA/medical coding with citation verification, tiebreaker, human escalation |
| **Model Guardrails** | Auto-detects embedding/vision/audio models, warns before chat, filters incompatible models from training |
| **Comprehensive Docs** | Built-in pitch deck, technical spec, user manual, API reference, and sales document in the sidebar |

---

## Quick Start

### Platform support

The platform runs on **macOS (Apple Silicon)** and **Linux** from a single codebase:

| | Training/inference backend | Serving |
|---|---|---|
| **macOS (Apple Silicon)** | MLX (Metal) | Ollama or in-app inference server (:7200) |
| **Linux / non-Mac** | HuggingFace + PyTorch (CPU, or CUDA if present) | in-app inference server (:7200) |

The backend is selected automatically at runtime (`mlx_available()` — true only with working Apple Metal). `requirements.txt` uses environment markers so platform-native deps (MLX) install **only** on macOS and never block a Linux install. GRPO is currently MLX-only and reports a clear "not available" message on Linux.

### Two ways to deploy

| Mode | Command | Backend |
|---|---|---|
| **Native** | `make serve` | **MLX/Metal on Apple Silicon**, HuggingFace/CPU on Linux |
| **Docker** | `make docker-up` (or `docker compose up -d --build`) | **HuggingFace/CPU on every host** |

Both start the web UI (`:7100`) and inference server (`:7200`) together. Your
fine-tuned models always appear in the chat picker (served by `:7200`).

**Ollama (base-model chat) with Docker — your choice:**

| | Command | Ollama |
|---|---|---|
| **Host Ollama** (default) | `make docker-up` | uses an Ollama running on your machine |
| **Bundled Ollama** (self-contained) | `make docker-up-ollama` | runs Ollama in a container and auto-pulls `qwen2.5:0.5b` |

Pull a different/extra set into the bundled Ollama with
`OLLAMA_PULL_MODELS="qwen2.5:0.5b qwen2.5:7b" make docker-up-ollama`. If no Ollama is
reachable, base models simply don't list — fine-tuned models still do.

> ⚠️ **Docker never uses MLX/Metal — even on a Mac.** Docker containers run a Linux
> VM with no access to Apple's Metal GPU, so a container always uses the CPU backend.
> **On Apple Silicon, use `make serve` (native) for MLX/Metal acceleration**; use
> Docker only when you want a portable CPU deployment (e.g. on Linux/servers).

### Prerequisites

- **macOS (Apple Silicon)** *or* **Linux** (x86_64 / arm64)
- **Python 3.12+** (recommended: 3.13 via [mise](https://mise.jdx.dev))
- **[Ollama](https://ollama.com)** — `brew install ollama` (macOS) / [install script](https://ollama.com/download) (Linux)
- **Node.js 22+** (optional, for BigSet dataset generation)
- **Linux/CPU tip:** install the lighter CPU PyTorch wheel first — `pip install torch --index-url https://download.pytorch.org/whl/cpu`

### Install & Run

```bash
git clone https://github.com/t4tarzan/finetune-platform.git
cd finetune-platform

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

brew services start ollama
python ui/app.py
```

Open **http://localhost:7100** — Chat tab is ready, sidebar has docs.

### First Run (Quick Test)

```bash
# Verify MLX detects your GPU
python -c "import mlx.core as mx; print('Metal:', mx.metal.is_available())"

# Try training with the built-in example data (10 Q&A pairs)
curl -X POST http://localhost:7100/api/train/start \
  -H 'Content-Type: application/json' \
  -d '{"niche":"test-run","verified_data_path":"data/example_train.jsonl","base_model":"mlx-community/Qwen2.5-0.5B-Instruct-4bit","epochs":2,"batch_size":4,"lora_rank":4}'

# Start the inference server (port 7200)
curl -X POST http://localhost:7100/api/inference/start

# Load a model into the inference server
curl -X POST 'http://localhost:7100/api/inference/load?model_path=mlx-community/Qwen2.5-0.5B-Instruct-4bit&model_name=demo'

# Chat with it via OpenAI-compatible API
curl -X POST http://localhost:7200/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"What is 2+2?"}],"max_tokens":50}'
```

### PM2 (Production)

```bash
pip install -r requirements.txt
pm2 start pm2.config.json
pm2 status finetune-platform
# http://localhost:7100
```

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     Browser (localhost:7100)                  │
│  ┌─────────┐  ┌─────────┐  ┌────────────────────────────┐  │
│  │ Chat Tab│  │Train Tab│  │ Sidebar: Docs + Leaderboard│  │
│  └────┬────┘  └────┬────┘  └────────────────────────────┘  │
│       │            │                                         │
├───────┴────────────┴─────────────────────────────────────────┤
│                      FastAPI Backend (ui/app.py)              │
│  ┌────────────┐  ┌──────────────┐  ┌────────────────────┐   │
│  │ /api/chat  │  │ /api/train/* │  │ /api/leaderboard   │   │
│  │ Ollama API │  │ SSE Progress │  │ Benchmark JSON     │   │
│  └────────────┘  └──────┬───────┘  └────────────────────┘   │
│                          │                                    │
├──────────────────────────┴────────────────────────────────────┤
│                   Training Manager (pipeline/)                 │
│  ┌────────────────┐  ┌──────────────┐  ┌────────────────┐   │
│  │ Consensus      │  │ MLX LoRA     │  │ Eval Harness   │   │
│  │ Verification   │──▶ Worker       │──▶ + Leaderboard  │   │
│  │ (4 models)     │  │ (subprocess) │  │                │   │
│  └────────────────┘  └──────┬───────┘  └────────────────┘   │
│                              │                                 │
│  ┌───────────────────────────┴────────────────────────────┐   │
│  │          ~/.cache/huggingface/hub/ (base models)       │   │
│  │          models/adapters/ (LoRA weights)               │   │
│  │          Ollama (GGUF serving)                         │   │
│  └────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

- **Subprocess isolation**: Each training job spawns a fresh Python process to avoid ML context leaks and CUDA/MLX state contamination
- **SSE streaming**: Server-Sent Events with heartbeat keep-alive for real-time training progress — no polling
- **JSONL over stdin/stdout**: Worker communicates with parent process via newline-delimited JSON for simplicity and debuggability
- **Consensus gate**: Only data verified by 3+ independent models enters the training set — reduces hallucination propagation
- **Port range**: All services use ports **7000-7500** per project conventions

---

## Pipeline Modules

### `pipeline/config.yaml`
Central configuration file. Controls:
- Base model path (HuggingFace MLX identifier)
- Consensus models (4 architectures for verification)
- Training hyperparameters (LoRA rank, batch size, learning rate, epochs)
- Consensus thresholds (min agreement, confidence)
- Eval settings (test split, retrieval top-k)
- Port assignments

### `pipeline/consensus_verifier.py`
Multi-model data verification. Each training row is sent to 4 models in parallel:
- `deepseek/deepseek-v4-pro`
- `Qwen/Qwen3.6-Max-Preview`
- `moonshotai/Kimi-K2.7-Code`
- `claude-sonnet-4-6`

Only rows where ≥3 models agree with confidence ≥0.7 pass to training.

**Output:** `verified_train.jsonl`, `rejected_train.jsonl`, `consensus_report.json`

```python
from pipeline.consensus_verifier import ConsensusVerifier, DataPoint

verifier = ConsensusVerifier()
verified, rejected, report = verifier.verify(dataset)
```

### `pipeline/training_worker.py`
Subprocess entry point running in an isolated Python process. Handles:
1. Model loading (MLX, supports quantized 4-bit)
2. LoRA adapter configuration
3. Training loop with progress emission via stdout JSONL
4. Checkpoint saving on stop or completion

**Event types emitted:**
```json
{"event": "progress", "step": 1, "loss": 4.89, "lr": 0.0001, "epoch": 0.5, "tokens_per_sec": 2400}
{"event": "status", "phase": "loading_model", "message": "Loading..."}
{"event": "complete", "output_dir": "models/adapters/niche", "final_loss": 2.33}
{"event": "error", "message": "Error description"}
```

### `pipeline/training_manager.py`
Orchestrates the training lifecycle:
- Spawns worker subprocess
- Reads stdout JSONL → puts on asyncio queue for SSE consumers
- Manages state machine: `idle → loading_model → training → completed/error/stopped`
- Persists run history to `data/training_history.json`
- Handles stop signals via file-based IPC (touch stop_file to interrupt)

### `pipeline/eval_harness.py`
Benchmark runner with:
- Held-out test set evaluation
- Metrics: accuracy, grounding (source citation %), consistency (cross-run stability)
- `BenchmarkLeaderboard` — persists scores with baseline deltas
- Reports improvement thresholds for recursive loop convergence

### `pipeline/export_gguf.py`
Post-training export:
1. Loads base model + LoRA adapter
2. Merges weights
3. Saves as MLX safetensors
4. Optionally registers with Ollama via Modelfile

### `pipeline/recursive_loop.py`
Full automation orchestrator:
1. Generate dataset via BigSet or load local JSONL
2. Consensus verify all rows
3. Prepare training/validation split
4. Run baseline eval (iter 1 only)
5. Fine-tune with MLX LoRA
6. Export + register with Ollama
7. Eval fine-tuned model
8. Record delta in leaderboard
9. Analyze gaps → refine data description → repeat

**Stops when:** accuracy ≥95%, improvement <1% for 2 consecutive iterations, or max iterations reached.

### `pipeline/train_qlora.py`
Data preparation + fine-tuning wrapper. Converts `{question, reference_answer}` JSONL into MLX-LoRA compatible format.

### `pipeline/data_store.py`
DuckDB-backed columnar storage layer. Replaces flat JSONL with SQL-powered data management:
- Import JSONL, CSV, or Parquet into DuckDB tables
- Export training sets via SQL queries
- Register customer database schemas for automated pipeline setup
- Auto-generate training SQL from customer schema (join tables → create QA pairs)

```python
from pipeline.data_store import DataStore
ds = DataStore("data/training.db")
ds.import_parquet("customer_billing.parquet")
ds.generate_training_set(
    "SELECT question, answer FROM billing_codes WHERE domain = 'icd10'",
    output_path="data/train.jsonl"
)
```

### `pipeline/inference_server.py`
Persistent MLX inference server with OpenAI-compatible API (runs on port 7200):
- `POST /v1/chat/completions` — chat completions (model stays hot in memory)
- `GET /v1/models` — list loaded models
- `POST /api/manage/load` — load a model by HF path
- Multiple models can stay hot simultaneously (up to 4 with 128GB)
- Manage from the main UI sidebar or via API

### `pipeline/model_discovery.py`
HuggingFace search agent that prevents unnecessary training:
1. Parses niche description → extracts keywords + task type
2. Searches HF Hub by pipeline_tag and keyword match
3. Downloads and evaluates each candidate locally via MLX
4. Ranks by accuracy vs parameter count (efficiency frontier)
5. Recommends: **use-as-is** (skip training), **fine-tune**, or **skip**

### `pipeline/grpo_trainer.py`
GRPO (Group Relative Policy Optimization) with binary LLM judge reward:
- Generates N completions per prompt (default: 4)
- LLM judge scores each as 1 (correct) or 0 (incorrect)
- Advantages normalized within group
- Policy updated with clipping (ε=0.2) + KL penalty (β=0.04)
- Runs as isolated subprocess (same pattern as training_worker.py)

---

## API Reference

### Chat & Models

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/models` | List all models with type detection (chat/embedding/vision) |
| GET | `/api/models/validate?model=` | Validate model type for chat (rejects embedding models) |
| GET | `/api/chat?message=&model=` | Chat with a model |

### Inference Server (port 7200)

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/inference/start` | Start the inference server |
| POST | `/api/inference/stop` | Stop it |
| GET | `/api/inference/status` | Check if running, models loaded |
| POST | `/api/inference/load?model_path=` | Load a model into hot memory |
| POST | `/v1/chat/completions` | OpenAI-compatible chat (on port 7200) |
| GET | `/v1/models` | OpenAI-compatible model list (on port 7200) |

### Data Store

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/data/stats` | DuckDB stats (tables, row counts, file size) |
| POST | `/api/data/import?path=&domain=` | Import JSONL/CSV/Parquet |
| GET | `/api/data/query?sql=` | Run SQL against the data store |
| GET | `/api/data/schema?table=` | Describe table schema |
| POST | `/api/data/generate?sql=&output_path=` | Generate training set from SQL |
| POST | `/api/data/register-customer` | Register customer DB schema |

### Training (port 7100)

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/train/start` | Start training with full config |
| GET | `/api/train/status` | Current training state and latest metrics |
| GET | `/api/train/progress` | SSE stream of live training events |
| POST | `/api/train/stop?save=true` | Stop training (optionally save checkpoint) |
| GET | `/api/train/history?limit=20` | Past training runs |
| GET | `/api/train/niches` | Available domain datasets |

### Discovery & RL

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/discover` | HF model search + local eval + recommend |
| POST | `/api/grpo/start` | Start GRPO reinforcement learning |

### Export & Leaderboard

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/export?niche=&adapter_path=` | Merge adapter + register with Ollama |
| GET | `/api/leaderboard?niche=` | Benchmark data (optionally filtered) |

### Documentation

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/docs/v1` | Full platform documentation as structured tree

---

## Usage Examples

### 1. Interactive Training via UI

```bash
source .venv/bin/activate
python ui/app.py
# → http://localhost:7100 → Train tab
```

### 2. Automated Recursive Loop

```bash
python pipeline/recursive_loop.py \
  --niche-name "medical-coding" \
  --niche-desc "Medical billing codes and descriptions from CMS.gov" \
  --max-iterations 5 \
  --max-rows 100
```

### 3. Consensus Verification Only

```python
from pipeline.consensus_verifier import ConsensusVerifier, DataPoint

data = [
    DataPoint(id="1", question="...", reference_answer="...", context="..."),
]

verifier = ConsensusVerifier()
verified, rejected, report = verifier.verify(data)
print(f"Verification rate: {report['summary']['verification_rate']:.0%}")
```

### 4. API-Driven Training

```bash
curl -X POST http://localhost:7100/api/train/start \
  -H 'Content-Type: application/json' \
  -d '{
    "niche": "my-domain",
    "base_model": "mlx-community/Qwen2.5-7B-Instruct-4bit",
    "epochs": 3,
    "verified_data_path": "data/verified_train.jsonl"
  }'

# Stream progress
curl -N http://localhost:7100/api/train/progress
```

---

## Configuration

See `pipeline/config.yaml` for all settings:

```yaml
base_model: "mlx-community/Qwen2.5-7B-Instruct-4bit"

consensus_models:
  - "deepseek/deepseek-v4-pro"
  - "Qwen/Qwen3.6-Max-Preview"
  - "moonshotai/Kimi-K2.7-Code"
  - "claude-sonnet-4-6"

consensus:
  min_agree: 3
  confidence_threshold: 0.7

training:
  lora_rank: 16
  lora_alpha: 32
  learning_rate: 1e-4
  batch_size: 4
  epochs: 3
  max_seq_length: 2048

ports:
  chat_ui: 7100
```

---

## Hardware Notes

| Configuration | Model Size (4-bit) | Feasibility |
|--------------|-------------------|-------------|
| 16GB RAM | 3B-7B models | Works for small domains |
| 32GB RAM | 7B-14B models | Good for most use cases |
| 64GB RAM | 14B-32B models | Large domains with rich data |
| **128GB RAM (M5 Max)** | **70B-class models** | **Heavy reasoning + RAG + training cache simultaneously** |

The platform runs on any M-series Mac. With 128GB unified memory, you can load a 70B 4-bit model (~45GB), keep a TurboVec index (~4GB), and still have ~75GB for training data and cache.

**MLX is Apple-native** — no CUDA, no ROCm, no translation layer. It accesses the unified memory pool directly via Apple's Metal Performance Shaders.

---

## Docker

> **Note:** Docker on macOS cannot access the Apple Silicon GPU natively (MLX requires Metal).  
> Docker is provided for the API/UI server only. For actual training, run natively with `make run`.

```bash
# Build and start
docker compose up -d --build

# View logs
docker compose logs -f

# Stop
docker compose down
```

---

## Project Structure

```
finetune-platform/
├── pipeline/
│   ├── config.yaml              # Central configuration
│   ├── consensus_verifier.py    # Multi-model data verification + compliance adjudication
│   ├── data_store.py            # DuckDB columnar storage, Parquet/CSV/SQL import
│   ├── train_qlora.py           # Data preparation wrapper
│   ├── training_worker.py       # Training subprocess entry point
│   ├── training_manager.py      # Training orchestration + SSE streaming
│   ├── eval_harness.py          # Benchmark + leaderboard
│   ├── export_gguf.py           # Model export + Ollama registration
│   ├── model_discovery.py       # HuggingFace search + local eval agent
│   ├── grpo_trainer.py          # GRPO RL with binary LLM judge reward
│   ├── inference_server.py      # Persistent MLX inference (port 7200, OpenAI API)
│   └── recursive_loop.py        # Full automation orchestrator
├── ui/
│   └── app.py                   # FastAPI server + HTML frontend (30+ API routes)
├── data/                        # DuckDB database + example JSONL data
├── models/                      # LoRA adapters + exported models
├── benchmarks/                  # Leaderboard JSON
├── requirements.txt             # 16 Python dependencies
├── Makefile                     # install/run/test/clean commands
├── pyproject.toml               # Python project metadata
├── docker-compose.yml           # Container orchestration
├── pm2.config.json              # Production process manager config
├── vercel.json                  # Vercel static site config
├── .gitignore
├── LICENSE                      # MIT
└── README.md                    # This file
```

---

## Extending

### Adding a New Consensus Model

Edit `pipeline/config.yaml`:

```yaml
consensus_models:
  - "nvidia/nemotron-3-ultra-550b-a55b"  # Add any cmd --list-models entry
```

### Adding a New Training Metric

1. Add the metric to `training_worker.py` ProgressCallback's `on_train_loss_report`
2. Add it to the SSE event payload
3. Add it to the frontend metric grid in `ui/app.py`

### Custom Export Format

Extend `pipeline/export_gguf.py` — currently supports MLX safetensors + Ollama GGUF. Add `transformers` push-to-hub, llama.cpp direct export, etc.

---

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| `mlx.core` has no `metal` | MLX not installed properly | `pip install mlx mlx-lm` and verify `python -c "import mlx.core as mx; print(mx.metal.is_available())"` |
| Training stalls at "importing modules" | First run downloads model | Wait for download or pre-cache: `python -c "from mlx_lm import load; load('mlx-community/Qwen2.5-7B-Instruct-4bit')"` |
| Worker exits with `Dataset must have at least batch_size` | Validation set too small | Increase dataset size or reduce batch_size |
| Ollama model not appearing | Export step failed | Run export manually: `POST /api/export?niche=...&adapter_path=...` |
| Consensus models return errors | API rate limits | Reduce `consensus.max_retries` or use fewer models |

---

## Development

```bash
# Install dev extras
pip install ruff pytest

# Lint
make lint

# Test (requires M-series Mac)
make test
```

---

## License

MIT — use freely, modify, distribute. Attribution appreciated.

## Acknowledgements

- [MLX](https://github.com/ml-explore/mlx) — Apple's ML framework for Apple Silicon
- [Unsloth](https://github.com/unslothai/unsloth) — Reference architecture for training UI patterns
- [Ollama](https://ollama.com) — Local model serving
- [BigSet](https://github.com/tinyfish-io/bigset) — Dataset generation from natural language
- [TurboVec](https://github.com/RyanCodrai/turbovec) — Google TurboQuant vector compression
- [Command Code](https://commandcode.ai) — Multi-model API for consensus verification
