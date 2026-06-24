#!/bin/sh
# Launch both platform services with one command:
#   - inference server (:7200) — serves merged/fine-tuned models
#   - web UI           (:7100) — chat + training (foreground; keeps the process alive)
#
# Used both as the Docker entrypoint and as the native launcher (`make serve`).
#
# The training/serving backend is auto-selected at runtime by mlx_available():
#   - bare-metal on Apple Silicon (venv has the MLX packages) -> MLX / Metal GPU
#   - everywhere else, INCLUDING any Docker container (always Linux) -> HuggingFace/CPU
# This is why Macs that want MLX/Metal acceleration must run this NATIVELY
# (`make serve`), not via Docker — containers cannot access Apple's Metal GPU.
set -e
cd "$(dirname "$0")/.."

# Prefer the project venv (native installs); fall back to the system interpreter
# (the Docker image installs dependencies system-wide, with no venv).
if [ -x ".venv/bin/python" ]; then
  PY=".venv/bin/python"
else
  PY="$(command -v python3 || command -v python || true)"
fi
[ -n "$PY" ] || { echo "[serve] no python interpreter found on PATH" >&2; exit 1; }

# ── Seed bundled datasets + adapter into the (often volume-mounted) data/models
# dirs on first boot. The image bakes them to /app/_bundled (not overlaid by the
# volume); cp -n preserves anything a user already added. One unified seed path for
# both `docker compose` (named volumes) and Kubernetes (PVCs) — no init container.
if [ -d /app/_bundled ]; then
  mkdir -p data models
  cp -rn /app/_bundled/data/. data/ 2>/dev/null || true
  cp -rn /app/_bundled/models/. models/ 2>/dev/null || true
  echo "[serve] seeded bundled datasets + adapter into data/ and models/"
fi

: "${PORT:=7100}"
: "${INFERENCE_PORT:=7200}"

echo "[serve] interpreter: $PY"
echo "[serve] inference server -> :${INFERENCE_PORT}"
"$PY" pipeline/inference_server.py &
INF_PID=$!

echo "[serve] web UI -> :${PORT}"
"$PY" ui/app.py &
UI_PID=$!

# Forward shutdown signals to BOTH children, then exit when the UI exits. (Using a
# backgrounded UI + wait rather than `exec` so the inference server is also signalled
# on `docker stop`, instead of being left for the kill-timeout to SIGKILL.)
trap 'kill "$INF_PID" "$UI_PID" 2>/dev/null' TERM INT
wait "$UI_PID" || true
kill "$INF_PID" 2>/dev/null || true
