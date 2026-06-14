# Multi-stage: builder + runtime
# Note: MLX requires Apple Silicon (metal). Docker runs natively on macOS,
# but for GPU passthrough use `make run` instead of Docker.

FROM python:3.13-slim AS builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.13-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

WORKDIR /app
COPY --from=builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY . .

EXPOSE 7100

CMD ["python", "ui/app.py"]
