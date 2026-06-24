# Fine-Tuning Platform — container image.
#
# Containers run Linux, so inside Docker the platform uses the HuggingFace/PyTorch
# CPU backend on BOTH macOS (Docker Desktop) and Linux hosts — MLX/Metal is only
# reachable bare-metal on Apple Silicon (`make run`). requirements.txt gates the MLX
# packages behind a `sys_platform == 'darwin'` marker, so they are never installed
# in this Linux image. The same image builds for the host architecture (arm64 on
# Apple Silicon, amd64 on Intel/Linux) — no platform pin needed.

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/root/.cache/huggingface

WORKDIR /app

# Install the CPU build of PyTorch first. The default PyPI wheel on Linux pulls
# ~2.5GB of CUDA/nvidia libraries we can't use on CPU; the +cpu wheel is small and
# satisfies the `torch==2.12.0` pin in requirements.txt (so the next step won't
# re-fetch the CUDA build). CPU wheels exist for both amd64 and arm64.
COPY requirements.txt .
RUN pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements.txt

# Bake the base models into the HF cache so the appliance trains/serves with NO
# network (air-gapped). BAKE_MODELS=0 builds a lean, online image instead.
# Layered before COPY so it caches independently of app-code changes.
ARG BAKE_MODELS=1
ARG HF_BAKE="Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen2.5-1.5B-Instruct"
RUN if [ "$BAKE_MODELS" = "1" ]; then \
      for m in $HF_BAKE; do python -c "from huggingface_hub import snapshot_download; snapshot_download('$m')"; done ; \
    fi
# Runtime offline: never reach the network for model files.
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

COPY . .

# Generate the SRE observability tables (cards) + Q&A training set, then stage ALL
# bundled artifacts (data + the pre-trained adapter) under /app/_bundled — a path the
# data/models volumes do NOT overlay. serve.sh copies them into data/ and models/ on
# first boot, so the seed works identically for `docker compose` and Kubernetes
# (no init container needed).
RUN python scripts/gen_sre_tables.py && python scripts/gen_sre_qa.py \
 && python scripts/gen_sre_demo_split.py \
 && mkdir -p /app/_bundled \
 && cp -r /app/data   /app/_bundled/data \
 && cp -r /app/models /app/_bundled/models

# 7100 = web UI (chat + train) · 7200 = OpenAI-compatible inference server
EXPOSE 7100 7200

# Same launcher used natively (`make serve`); in the container there is no .venv,
# so it runs system Python -> HuggingFace/CPU backend.
CMD ["sh", "scripts/serve.sh"]
