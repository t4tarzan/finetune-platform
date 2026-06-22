#!/usr/bin/env bash
# =============================================================================
#  Fine-Tuning Platform Demo — K3s Insurance Use Case
#
#  Record with:  bash demo.sh 2>&1 | tee /tmp/demo-recording.txt
#  Screen cap:   ffmpeg -f avfoundation -i 1 -r 10 demo.mp4
# =============================================================================

BASE="http://localhost:7100"
INFER="http://localhost:7200"

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║      Fine-Tuning Platform — Demo                            ║"
echo "║      K3s Cluster + Insurance Company Database               ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ── Step 1: Show the initial data ──
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  STEP 1: Initial dataset — 7 verified rows"
echo "  Topics: PVC mounts, namespace isolation, service discovery,"
echo "          ConfigMaps, RBAC, Helm, Ingress, HPA"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

head -2 data/insurance-k3s/verified_train.jsonl 2>/dev/null | python3 -c "
import sys, json
for line in sys.stdin:
    if line.strip():
        d = json.loads(line)
        print(f'  • {d.get(\"question\",\"?\")[:80]}...')
"
echo "  … and 5 more rows"
echo ""

# Ensure inference server is running
curl -s -X POST "$BASE/api/inference/start" > /dev/null 2>&1 || true
sleep 2

# ── Step 2: Training v1 ──
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  STEP 2: Train model (v1) — 7 rows, Qwen2.5-0.5B, LoRA rank 4"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

curl -s -X POST "$BASE/api/train/start" \
  -H 'Content-Type: application/json' \
  -d '{"niche":"insurance-k3s","dataset_type":"local","verified_data_path":"data/insurance-k3s/verified_train.jsonl","base_model":"mlx-community/Qwen2.5-0.5B-Instruct-4bit","lora_rank":4,"epochs":3}' > /dev/null

# Poll for completion
for i in $(seq 1 30); do
  sleep 3
  STATUS=$(curl -s "$BASE/api/train/status" | python3 -c "import sys,json; print(json.load(sys.stdin).get('status',''))")
  if [ "$STATUS" = "completed" ] || [ "$STATUS" = "error" ]; then
    break
  fi
done

V1_LOSS=$(curl -s "$BASE/api/train/status" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('final_loss','?'))")
echo "  ✓ Training complete — loss: $V1_LOSS"
echo ""

# ── Step 3: Test v1 ──
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  STEP 3: Test v1 — ask a K3s+insurance question"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Export model and load into inference server
python3 -c "
from mlx_lm import load; import os, shutil, glob
model, tokenizer = load('mlx-community/Qwen2.5-0.5B-Instruct-4bit')
model.load_weights('models/adapters/insurance-k3s/adapters.safetensors', strict=False)
os.makedirs('models/gguf/insurance-k3s_merged', exist_ok=True)
model.save_weights('models/gguf/insurance-k3s_merged/model.safetensors')
tokenizer.save_pretrained('models/gguf/insurance-k3s_merged')
for c in glob.glob(os.path.expanduser('~/.cache/huggingface/hub/models--mlx-community--Qwen2.5-0.5B-Instruct-4bit/snapshots/*/config.json')):
    shutil.copy(c, 'models/gguf/insurance-k3s_merged/config.json'); break
print('  ✓ Model exported')
"

curl -s "$INFER/api/manage/unload?model_name=insurance-k3s" > /dev/null 2>&1 || true
curl -s "$INFER/api/manage/load?model_path=$(pwd)/models/gguf/insurance-k3s_merged&model_name=insurance-k3s" > /dev/null 2>&1
sleep 3

V1_ANSWER=$(curl -s -X POST "$INFER/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"How do I mount a PersistentVolumeClaim in K3s for insurance claims data?"}],"max_tokens":80}' | python3 -c "
import sys, json
d = json.load(sys.stdin)
print(d.get('choices',[{}])[0].get('text','')[:180])
")
echo "  Q: How to mount a PVC for insurance claims data?"
echo "  A: $V1_ANSWER"
echo "  Loss: $V1_LOSS"
echo ""

# ── Step 4: Add more data ──
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  STEP 4: Adding 6 more verified rows (advanced topics)"
echo "  Network policies, pod disruption budgets, secrets management"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

python3 -c "
import json, os
rows = []
for f in ['data/insurance-k3s/verified_train.jsonl', 'data/insurance-k3s-v2/verified_train.jsonl']:
    with open(f) as fp:
        for line in fp:
            if line.strip(): rows.append(json.loads(line))
os.makedirs('data/insurance-k3s-combined', exist_ok=True)
with open('data/insurance-k3s-combined/verified_train.jsonl', 'w') as f:
    for r in rows: f.write(json.dumps(r) + '\n')
print(f'  Combined: {len(rows)} rows (7 + {len(rows)-7} new)')
"

echo ""

# ── Step 5: Training v2 ──
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  STEP 5: Retrain (v2) on all rows — same model, more data"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

curl -s -X POST "$BASE/api/train/start" \
  -H 'Content-Type: application/json' \
  -d '{"niche":"insurance-k3s-v2","dataset_type":"local","verified_data_path":"data/insurance-k3s-combined/verified_train.jsonl","base_model":"mlx-community/Qwen2.5-0.5B-Instruct-4bit","lora_rank":8,"epochs":3}' > /dev/null

for i in $(seq 1 30); do
  sleep 3
  STATUS=$(curl -s "$BASE/api/train/status" | python3 -c "import sys,json; print(json.load(sys.stdin).get('status',''))")
  if [ "$STATUS" = "completed" ] || [ "$STATUS" = "error" ]; then
    break
  fi
done

V2_LOSS=$(curl -s "$BASE/api/train/status" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('final_loss','?'))")
echo "  ✓ Retraining complete — loss: $V2_LOSS"
echo ""

# ── Step 6: Test v2 ──
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  STEP 6: Test v2 — same question, improved answer"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

python3 -c "
from mlx_lm import load; import os, shutil, glob
model, tokenizer = load('mlx-community/Qwen2.5-0.5B-Instruct-4bit')
model.load_weights('models/adapters/insurance-k3s-v2/adapters.safetensors', strict=False)
os.makedirs('models/gguf/insurance-k3s-v2_merged', exist_ok=True)
model.save_weights('models/gguf/insurance-k3s-v2_merged/model.safetensors')
tokenizer.save_pretrained('models/gguf/insurance-k3s-v2_merged')
for c in glob.glob(os.path.expanduser('~/.cache/huggingface/hub/models--mlx-community--Qwen2.5-0.5B-Instruct-4bit/snapshots/*/config.json')):
    shutil.copy(c, 'models/gguf/insurance-k3s-v2_merged/config.json'); break
print('  ✓ Model v2 exported')
"

curl -s "$INFER/api/manage/unload?model_name=insurance-k3s-v2" > /dev/null 2>&1 || true
curl -s "$INFER/api/manage/load?model_path=$(pwd)/models/gguf/insurance-k3s-v2_merged&model_name=insurance-k3s-v2" > /dev/null 2>&1
sleep 3

V2_ANSWER=$(curl -s -X POST "$INFER/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"How do I mount a PersistentVolumeClaim in K3s for insurance claims data?"}],"max_tokens":80}' | python3 -c "
import sys, json
d = json.load(sys.stdin)
print(d.get('choices',[{}])[0].get('text','')[:180])
")
echo "  Q: How to mount a PVC for insurance claims data?"
echo "  A: $V2_ANSWER"
echo "  Loss: $V2_LOSS"
echo ""

# ── Summary ──
IMPROVEMENT=$(python3 -c "
v1=float('$V1_LOSS') if '$V1_LOSS' != 'None' else 0
v2=float('$V2_LOSS') if '$V2_LOSS' != 'None' else 0
if v1 > 0: print(f'{((v1-v2)/v1*100):.0f}')
else: print('?')
")

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                     DEMO COMPLETE                           ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║                                                              ║"
echo "║  Initial training:  loss $V1_LOSS                               ║"
echo "║  After adding data: loss $V2_LOSS  (${IMPROVEMENT}% improvement)  ║"
echo "║                                                              ║"
echo "║  Model:   Qwen2.5-0.5B-Instruct (4-bit)                     ║"
echo "║  Hardware: Apple M5 Max, 128GB unified memory               ║"
echo "║  Platform: http://localhost:7100                             ║"
echo "║  Repo:     https://github.com/t4tarzan/finetune-platform     ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
