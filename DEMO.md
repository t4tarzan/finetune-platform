# SRE Incremental Fine-Tuning Demo

Reproduces the headline demo: train a small local model on SRE/Kubernetes
incident data, then **retrain (continue) it on more data** and watch both the
**loss curve drop further** and the **answer get sharper** — all offline on
Apple Silicon (MLX) or CPU (HuggingFace fallback).

## What ships in the repo vs. what you build

- **Ships:** the two datasets (`data/sre-pods/dataset_v1.jsonl` = 100 rows,
  `dataset_v2.jsonl` = 150 rows = same 100 + 50 deeper rows) and the generator
  (`scripts/gen_sre_dataset.py`) that recreates them deterministically.
- **You build (locally):** the fine-tuned models. The trained adapters + merged
  models are several GB — too large for git — so `scripts/run_sre_demo.sh`
  rebuilds them by training (seeded, so results match closely). First run also
  downloads the base model `mlx-community/Qwen2.5-1.5B-Instruct-4bit` from
  Hugging Face (~1 GB, cached).

## Run it

```bash
# 1. Install + start the platform (UI :7100 + inference :7200)
pip install -r requirements.txt
make serve            # or: python ui/app.py    (Apple Silicon: brew install ollama; brew services start ollama)

# 2. In another terminal, rebuild the two models and chat the hero question
bash scripts/run_sre_demo.sh
```

That trains `sre-pods` (100 rows, from base) → exports/serves it → retrains
`sre-pods-v2` (150 rows, **continuing from the sre-pods adapter**) → exports it,
then prints the before/after answers to the hero question.

## Do it by hand in the UI (what the video shows)

1. **Train tab** → niche `sre-pods`, dataset `data/sre-pods/dataset_v1.jsonl`,
   click **Use This Dataset**. Base model **Qwen2.5-1.5B (4-bit)**, rows `120`.
   **Start Training** → watch the loss curve fall (~3.4 → ~0.03).
2. Click **⬆ Export & Serve Fine-Tuned Model**.
3. **Chat tab** → pick `sre-pods (fine-tuned)`, ask the hero question →
   a generic CrashLoopBackOff answer.
4. **Train tab** → niche `sre-pods-v2`, dataset `dataset_v2.jsonl`,
   **Use This Dataset**, set **Continue from fine-tuned → sre-pods**, rows `200`.
   **Start Training** → the loss curve starts low (~0.03, not 3.4) and drops to
   ~0.018.
5. **Export & Serve**, then **Chat** with `sre-pods-v2` → a sharper answer that
   nails the node-drain root cause + a real preventive checklist (PDB, graceful
   shutdown, replica spread).

**Hero question:**
> Our payments-api pods keep entering CrashLoopBackOff right after a node drain /
> cluster autoscaler scale-down. What is the most likely root cause, and what
> preventive measures stop it from happening again?

## Record the video

```bash
npm install                              # installs Playwright
npx playwright install chromium          # one-time browser download
node demo-record-sre.js demo-sre.mp4     # ~3 min; trains live, captions on-screen
```

> Tip: for a clean recording, remove prior runs first:
> `rm -rf models/adapters/sre-pods* models/gguf/sre-pods*_merged`

## Regenerate the datasets (optional)

```bash
python scripts/gen_sre_dataset.py        # rewrites dataset_v1.jsonl + dataset_v2.jsonl
```
