#!/usr/bin/env bash
#
# One-command install of the Fine-Tuning Platform on an EKS (or any) cluster using
# the prebuilt PUBLIC image (ghcr.io/t4tarzan/finetune-platform:latest, amd64).
# No docker build, no pull secret.
#
# Prereqs: kubectl (pointed at your cluster), helm, a StorageClass (default gp3),
# an amd64 node, and egress to ghcr.io (+ huggingface.co for the base model).
#
# Usage:
#   bash scripts/install-eks.sh
#   NS=ml STORAGE_CLASS=gp3 BASE_MODEL=qwen2.5:1.5b bash scripts/install-eks.sh
set -euo pipefail

NS="${NS:-finetune}"
STORAGE_CLASS="${STORAGE_CLASS:-gp3}"
BASE_MODEL="${BASE_MODEL:-qwen2.5:0.5b}"
CHART="$(cd "$(dirname "$0")/.." && pwd)/charts/finetune-platform"

echo "==> Installing release 'finetune-platform' into namespace '$NS' (storageClass=$STORAGE_CLASS)"
helm upgrade --install finetune-platform "$CHART" \
  --namespace "$NS" --create-namespace \
  --set persistence.storageClass="$STORAGE_CLASS" \
  --set ollama.enabled=true

echo "==> Waiting for rollout (models are baked into the image — no download)"
kubectl -n "$NS" rollout status deploy/finetune-platform --timeout=600s
echo "   (first boot seeds 43 tables into DuckDB + merges the bundled sre-assistant adapter)"

cat <<EOF

==> Done. Open the UI:
    kubectl -n $NS port-forward svc/finetune-platform 7100:7100
    # then browse http://localhost:7100

Out of the box: preset cards, "Query data" (text-to-SQL), a fine-tuned sre-assistant,
and the train -> append -> retrain loop. Optional base-model chat:
    kubectl -n $NS exec -c ollama deploy/finetune-platform -- ollama pull $BASE_MODEL

Expose:  helm upgrade finetune-platform "$CHART" -n $NS --reuse-values --set service.type=LoadBalancer
Sub-path ingress (nginx): add --set ingress.enabled=true --set ingress.className=nginx --set basePath=/finetune-platform
EOF
