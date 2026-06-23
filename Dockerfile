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

COPY . .

# Generate the bundled SRE observability tables into the image (data/sre-tables/*.csv).
# Done at build (not committed) so the appliance ships with the data; the chart's init
# container seeds these into the data volume on first boot for the chat's preset cards.
RUN python scripts/gen_sre_tables.py

# 7100 = web UI (chat + train) · 7200 = OpenAI-compatible inference server
EXPOSE 7100 7200

# Same launcher used natively (`make serve`); in the container there is no .venv,
# so it runs system Python -> HuggingFace/CPU backend.
CMD ["sh", "scripts/serve.sh"]
