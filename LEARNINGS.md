# Learnings & Operational Notes

Hard-won, verified knowledge about running this platform — especially the
**Linux / Docker / CPU** path, which is not the project's native target. The
upstream project is **Apple-Silicon / MLX-native**; everything below documents how
it actually behaves when run elsewhere, the cross-platform design that makes that
possible, and the gotchas found along the way.

> Scope: this complements `README.md` (how to run) and `INSTALL-EKS.md` (k8s). It is
> the "why it works this way / what surprised us" reference.

---

## 1. The two backends (the single most important concept)

The platform runs on two completely different stacks, chosen **at runtime** by
`pipeline/training_manager.py::mlx_available()` (true only on Apple Silicon with
working Metal):

| | **MLX backend** | **HuggingFace / CPU backend** |
|---|---|---|
| Where | macOS, Apple Silicon, **native** (not Docker) | Linux, **incl. all Docker/containers** (any host, even a Mac) |
| Trainer | `pipeline/training_worker.py` (MLX LoRA) | `pipeline/training_worker_hf.py` (PyTorch+PEFT) |
| Export | `pipeline/export_gguf.py` | `pipeline/export_hf.py` |
| Base precision | **4-bit quantized** (mlx-community models) | **full precision fp32** |
| Serving | Ollama (GGUF) | in-app inference server on **:7200** |
| Base chat | Ollama | Ollama (host or bundled) |

**Containers are always Linux**, so a container *always* uses the HF/CPU backend —
MLX/Metal is unreachable inside Docker even on a Mac host. For MLX acceleration you
must run bare-metal on Apple Silicon.

**Cross-platform invariants to preserve** (breaking these breaks one OS):
- MLX is imported **lazily** inside the functions that need it, never at module top,
  so `pipeline/*` imports cleanly on Linux. `prepare_training_data` is pure Python.
- All backend forks go through `mlx_available()` — don't hardcode a path.
- MLX deps in `requirements.txt` are gated with `; sys_platform=='darwin'`.
- A harmless `mlx`/`mlx_lm` shim lets the server import on Linux; the real MLX worker
  is never invoked there because of the dispatch.

---

## 2. Why fp32 on CPU, why 4-bit on MLX, why not just reuse Ollama?

Three questions that come up constantly:

**Why does Linux/CPU download fp32 from HuggingFace instead of reusing the Ollama model?**
Because the Ollama model **can't be fine-tuned**. Ollama serves **GGUF** (a llama.cpp
*inference* format) — no autograd, no trainable params, not loadable by PEFT/transformers.
LoRA training needs PyTorch/safetensors weights so it can freeze the base and backprop
into the adapter. And training a *4-bit* base (QLoRA) needs `bitsandbytes`+CUDA, which
doesn't exist on CPU; fp16/bf16 CPU ops are slow/flaky. So on CPU the only stable path
is **fp32 base + LoRA adapter**. Ollama = inference only; HF fp32 = required for training.

**Why does Mac/MLX use 4-bit instead of fp32?**
Efficiency the hardware makes practical. MLX natively trains a LoRA adapter on a
**frozen 4-bit-quantized base** on Metal; a 7B model fits in ~4 GB of unified memory
(fp32 would be ~28 GB). CPU can't do efficient 4-bit, so it falls back to fp32 with
small models only (0.5–1.5B). Same idea (freeze base, train adapter), different base
precision per hardware.

**The "(4-bit)" labels in the old UI were misleading on Linux** — the HF backend loads
`dtype=torch.float32` everywhere (`training_worker_hf.py`, `export_hf.py`,
`inference_server.py`). 4-bit only applies to native Apple-Silicon. The base-model
picker is now **backend-aware** (see §8).

---

## 3. Model naming — three different id schemes for the "same" model

This trips everyone up. `Qwen2.5-0.5B` shows up under three names:

| Name | What it is | Used for |
|---|---|---|
| `qwen2.5:0.5b` | **Ollama** tag (GGUF, Q4_K_M, ~397 MB) | base chat via Ollama |
| `Qwen/Qwen2.5-0.5B-Instruct` | **HuggingFace** repo id (fp32 safetensors, ~1 GB) | the CPU **trainer** downloads & fine-tunes this |
| `mlx-community/Qwen2.5-*-4bit` | **MLX** 4-bit quantized | MLX training on Apple Silicon only |

`pipeline/training_worker_hf.py::resolve_model_id()` is the bridge on Linux: it maps
any `mlx-community/*`, any `ollama:tag`, or empty → `CPU_DEFAULT_MODEL`
(`Qwen/Qwen2.5-0.5B-Instruct`), and **passes through a real HF id unchanged**. So on
Linux, selecting any MLX option in the old dropdown silently trained 0.5B — the source
of much confusion (fixed in §8).

CPU constants: `CPU_DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"`,
`CPU_MAX_SEQ_LEN = 1024` (config's 2048 default is capped here — long contexts are
truncated, which keeps CPU steps tractable).

---

## 4. Dataset schema — what trains and what gets silently dropped

The trainer ultimately consumes line-delimited JSON (`.jsonl`), normalized by
`pipeline/train_qlora.py::prepare_training_data` into `{prompt, completion}`. The input
file accepts **two row shapes**:

| Shape | Required fields | Optional |
|---|---|---|
| Q&A (canonical) | `question`, `reference_answer` | `context` |
| Raw LM pair | `prompt`, `completion` | — |

**Rows missing both pairs are silently skipped** — not an error. So a CSV/JSONL with
the wrong column names trains on *zero* rows. (e.g. the `data/sre-pods/*` demo data has
`question`/`reference_answer`/`context` → all rows train; a generic table like
`Model, License, Downloads` would be dropped entirely.)

**CSV upload** (`POST /api/data/upload-csv?niche=<name>`, Train-tab "Upload CSV"):
reads the raw CSV body (stdlib `csv`, no dependency), detects the schema
case-insensitively, converts to `data/<niche>/verified_train.jsonl`, and feeds the
normal flow. Arbitrary CSVs need a `question`/`reference_answer` (or `prompt`/`completion`)
header, or an explicit column mapping — there is no inherent Q&A in a generic table.

---

## 5. Verified capability matrix (with a pre-existing JSONL, on Linux/Docker/CPU)

All confirmed end-to-end on the HF/CPU backend:

- **Train** → `pipeline/training_worker_hf.py` (LoRA, fp32, q_proj/v_proj).
- **Export** → `pipeline/export_hf.py` merges the adapter and serves the merged model
  on **:7200** (PEFT writes `adapter_model.safetensors` + `base_model.txt` +
  `training_metadata.json`).
- **Serve / chat** → `:7200` (OpenAI-compatible) for fine-tuned models; **Ollama** for
  base models. The chat route sends inference-provider models to `:7200` *before*
  `validate_model` so niche names aren't blocked, and lazy-loads merged dirs on first chat.
- **Evaluate + leaderboard delta** → `/api/evaluate` serves base → evals (baseline) →
  unloads → serves fine-tuned → evals → records iteration with per-metric delta. The
  baseline base-model is resolved from the adapter's `base_model.txt` (apples-to-apples).
- **Recursive loop** → `pipeline/recursive_loop.py` dispatches per backend.
- **Incremental retraining** → continue from a prior adapter (see §6).

Out of scope / disabled: **BigSet** auto-generation and **consensus** verification
(need external API keys, see §9); **GRPO** (MLX-only by design).

---

## 6. Incremental retraining (continue from a previous adapter)

"Fine-tune on top of v1 with more data" = the `resume_adapter` param. This was wired
only for MLX upstream; we **added it to the HF/CPU backend**:
- `training_worker_hf.py`: if `resume_adapter` points at a dir with `adapter_config.json`,
  load it with `PeftModel.from_pretrained(model, resume_adapter, is_trainable=True)` and
  keep training — onto the **exact base it was trained on** (read from `base_model.txt`),
  instead of a fresh `get_peft_model`.
- `/api/adapters` now detects **both** `adapters.safetensors` (MLX) and
  `adapter_model.safetensors` (HF) so HF adapters appear in the "Continue from
  fine-tuned" picker.

**How to prove a v2 truly continued from v1 (not silently from base)** — three angles
that must agree:
1. **Log**: the run logs `phase: lora_resumed` / "Continuing training from adapter: …"
   and the init line says "…resuming from …".
2. **Loss/grad**: a resumed run *opens* at a markedly lower loss + smaller grad_norm
   than a from-base run on the **same data** (a model can't start low on data it's
   never seen). Controlled A/B observed: from-base step-1 loss ≈ 3.79 (grad ~4.7) vs
   resume-from-v1 ≈ 2.92 (grad ~2.0).
3. **Metadata**: both adapters' `base_model.txt` confirm the same base, so it's strictly
   the *adapter* being continued.

---

## 7. Ports, processes, deployment

- **:7100** — web UI (Chat + Train), `ui/app.py` (FastAPI).
- **:7200** — in-app OpenAI-compatible inference server, `pipeline/inference_server.py`.
- **:11434** — Ollama (host or bundled container).
- `scripts/serve.sh` runs both :7100 and :7200; healthcheck must probe **both**.

**Docker** (`docker-compose.yml`): one image runs on macOS and Linux (no `platform:`
pin). Ollama is a **choice**: default = host Ollama via `host.docker.internal`; or the
bundled `ollama` profile (`make docker-up-ollama`) which auto-pulls
`OLLAMA_PULL_MODELS`. Volumes persist `data/`, `models/`, `benchmarks/`, `logs/`, and
the HF cache.

**Deploy/update rule (learned the hard way):** source is **baked into the image**
(`COPY . .`), only those dirs are volume-mounted. So a code change needs
`docker compose up -d --build` — **not** `docker cp` (ephemeral, lost on next rebuild).
`docker cp` is fine only for a throwaway test against the live container.

---

## 8. UI fixes made for the Linux/Docker deployment

Several UI elements assumed the MLX/Mac world and misbehaved on Linux:

- **Backend-aware base-model picker** — new `GET /api/platform` reports the active
  backend + the base models valid for it (HF fp32 ids on CPU, MLX 4-bit on Mac), with a
  precision-accurate hint. Replaces the static MLX-only `<select>` whose options all
  collapsed to one 0.5B model on Linux.
- **"Use This Dataset" fallback** — the new bundled-dataset dropdown only copies its
  value into the path field on a `change` event; re-selecting the current option fires
  none, so the button errored with "Enter a dataset path first." Now `useLocalDataset()`
  falls back to the dropdown's current selection.
- **Inference "Load Model" bar** (sidebar) had two bugs: (a) default was an MLX 4-bit id
  the HF `:7200` server can't load (now backend-aware), and (b) `/api/inference/load` and
  `/unload` did a **GET** against POST-only `:7200` endpoints → "405 Method Not Allowed"
  (now proper POST). This bar loads any HF id or local merged dir into `:7200` for
  serving/chat; it is **not** a training base (training base is set in the Train tab).

---

## 9. API keys — what's needed (nothing, for the core flow)

- **No `HF_TOKEN`** is configured or required — HF downloads are **anonymous**
  (you'll see a "set HF_TOKEN for higher rate limits" warning; it still works). A token
  is only needed for **gated/private** models or to avoid anonymous rate limits.
- `.env` (host) has `OPENROUTER_API_KEY`, `TINYFISH_API_KEY`, `BIGSET_LOCAL_WORKSPACE_ID`
  — these are **only** for the optional **BigSet auto-generate / consensus** dataset
  feature (disabled). The container doesn't even load `.env` (no `env_file` in compose),
  so the running deployment has **zero keys** — proof the whole local pipeline (train →
  export → serve → chat → evaluate → incremental retrain) runs **fully offline**.

---

## 10. Gotchas worth remembering

- **Ollama crashes on our converted models** (exit status 2) on this Linux box; stock
  Ollama models work. That's *why* fine-tuned models are served via the in-app `:7200`
  server, not Ollama, on Linux.
- **`pyyaml==6.0` fails to build** on Python 3.12 (Cython 3 error) → use `6.0.3`.
- **Install Torch from the CPU index** (`--index-url .../whl/cpu`) or pip pulls ~2 GB of
  useless CUDA packages.
- **Worker subprocess stderr must not be a PIPE** that's only drained on error — a chatty
  first run (HF download bars) can fill the 64 KB pipe buffer and **deadlock**. Redirect
  worker stderr to a file (done in `recursive_loop.py`).
- **`.venv/bin/python` may not exist** in the image → fall back to `sys.executable` when
  spawning workers.
- **0.5B + few epochs on CPU undertrains** — loss plateaus ~2.4 and answers improve in
  *style* but not always *content*. For a convincing before/after demo, train enough
  epochs (memorization) or accept the limitation. CPU step time scales with sequence
  length; long contexts (e.g. kubectl logs) dominate.

---

## 11. Git / fork workflow (this clone's convention)

- `origin` **and** `upstream` both point to **`01rmachani/finetune-platform`** by design.
  Do **not** re-add `t4tarzan` as a remote.
- The fork's parent is `t4tarzan/finetune-platform`. "Sync the fork" = pull upstream's
  new commits in. The fork is **diverged** (ahead with local work), so `gh repo sync`
  refuses and `--force` would wipe local commits.
- Correct sync: one-off `git fetch https://github.com/t4tarzan/finetune-platform.git main`
  (no remote added) → `git merge FETCH_HEAD` → resolve → push to `origin`. The GitHub
  compare API may show a stale `behind_by`; the authoritative check is an **empty**
  upstream-only commit list.
- Pushing local work to `t4tarzan` is done via a **cross-fork PR**, manually.

---

## 12. Recording a demo (optional)

`demo-record-sre-flow.js` (Playwright) records the full base → train v1 → export →
chat → continue-train v2 → export → chat flow to an `.mp4` (needs `ffmpeg` + a Node
Playwright+Chromium install). Training waits are real (polled), so set generous
timeouts; on CpU, v2 with long contexts can take tens of minutes. See §10 re: 0.5B
undertraining if the answer improvement looks weak.
