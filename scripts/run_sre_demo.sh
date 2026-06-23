#!/usr/bin/env bash
#
# Reproduce the SRE / Kubernetes incremental fine-tuning demo end-to-end.
#
# This rebuilds the exact two models the demo uses (they're too large to ship in
# git, so we train them here — training is seeded, so results match closely):
#
#   1. train  sre-pods      on data/sre-pods/dataset_v1.jsonl   (100 rows, from base)
#   2. export sre-pods      (fuse adapter -> standalone served model)
#   3. train  sre-pods-v2   on data/sre-pods/dataset_v2.jsonl   (150 rows,
#                            CONTINUING from the sre-pods adapter — incremental)
#   4. export sre-pods-v2
#
# Then chat the "hero" question against each to see the answer improve, and
# optionally record the video with:  node demo-record-sre.js demo-sre.mp4
#
# Prereq: the platform must be running on :7100 (make serve) and, on Apple
# Silicon, the MLX base model downloads automatically on first train.
#
# Usage:  bash scripts/run_sre_demo.sh
set -euo pipefail

BASE="${BASE:-http://localhost:7100}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

HERO='Our payments-api pods keep entering CrashLoopBackOff right after a node drain / cluster autoscaler scale-down. What is the most likely root cause, and what preventive measures stop it from happening again?'

say() { printf '\n\033[1;36m== %s ==\033[0m\n' "$*"; }

require_server() {
  if ! curl -sf -o /dev/null "$BASE/"; then
    echo "ERROR: platform not reachable at $BASE — start it first (make serve)." >&2
    exit 1
  fi
}

# Poll /api/train/status until the run is no longer running.
wait_train() {
  local label="$1"
  while true; do
    local st
    st=$(curl -s "$BASE/api/train/status" | python3 -c "import sys,json;print(json.load(sys.stdin).get('status',''))" 2>/dev/null || echo "")
    case "$st" in
      completed) curl -s "$BASE/api/train/status" | python3 -c "import sys,json;d=json.load(sys.stdin);print('  %s done — final_loss=%s, steps=%s'%('$label',d.get('final_loss'),d.get('total_steps')))"; return 0 ;;
      error|stopped) echo "  $label ended: $st" >&2; return 1 ;;
      *) printf '.' ; sleep 3 ;;
    esac
  done
}

train() {  # niche  dataset_path  max_rows  resume_adapter
  curl -s -X POST "$BASE/api/train/start" -H "Content-Type: application/json" -d "$(python3 - "$@" <<'PY'
import json,sys
niche,path,rows,resume=sys.argv[1:5]
print(json.dumps({
  "niche":niche,"dataset_type":"local","verified_data_path":path,
  "base_model":"mlx-community/Qwen2.5-1.5B-Instruct-4bit",
  "lora_rank":16,"lora_alpha":32,"learning_rate":1e-4,
  "batch_size":4,"epochs":3,"max_seq_length":2048,
  "max_rows":int(rows),"resume_adapter":resume,
}))
PY
)" | python3 -c "import sys,json;d=json.load(sys.stdin);print('  started',d.get('niche'),d.get('status'),'resume=' + (d.get('params',{}).get('resume_adapter') or 'none'))"
}

export_model() {  # niche
  curl -s -X POST "$BASE/api/export?niche=$1&adapter_path=models/adapters/$1" \
    | python3 -c "import sys,json;d=json.load(sys.stdin);print('  export',d.get('status',d))"
}

chat() {  # model
  curl -s -G "$BASE/api/chat" --data-urlencode "model=$1" --data-urlencode "message=$HERO" --max-time 300 \
    | python3 -c "import sys,json;d=json.load(sys.stdin);print((d.get('response') or str(d))[:700])"
}

require_server

say "1/4  Train v1 (sre-pods, 100 rows, from base 1.5B)"
train sre-pods data/sre-pods/dataset_v1.jsonl 120 ""
wait_train v1

say "2/4  Export v1"
export_model sre-pods

say "3/4  Retrain v2 (sre-pods-v2, 150 rows, CONTINUING from v1 adapter)"
train sre-pods-v2 data/sre-pods/dataset_v2.jsonl 200 "models/adapters/sre-pods"
wait_train v2

say "4/4  Export v2"
export_model sre-pods-v2

say "Hero question — v1 (sre-pods)"
chat sre-pods
say "Hero question — v2 (sre-pods-v2, retrained)"
chat sre-pods-v2

say "Done. Models built & served. Record the video with:  node demo-record-sre.js demo-sre.mp4"
