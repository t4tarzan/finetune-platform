.PHONY: install run serve train export clean docker-up docker-up-ollama docker-down

VENV = .venv/bin

install: $(VENV)/python
	$(VENV)/pip install -r requirements.txt

$(VENV)/python:
	python3 -m venv .venv
	$(VENV)/pip install --upgrade pip

# Web UI only (inference server started on demand from the UI).
run: install
	$(VENV)/python ui/app.py

# Full native deploy: inference server (:7200) + web UI (:7100) in one command.
# On Apple Silicon this uses the MLX/Metal backend — use this (not Docker) for
# GPU acceleration on Mac; on Linux it uses the HuggingFace/CPU backend.
serve: install
	./scripts/serve.sh

train: install
	$(VENV)/python pipeline/recursive_loop.py --niche-name "$(niche)" --niche-desc "$(desc)" --max-iterations $(iters)

export: install
	$(VENV)/python pipeline/export_gguf.py

clean:
	rm -rf data/*/run_*/
	rm -rf models/adapters/*
	rm -rf models/gguf/*
	rm -rf benchmarks/*.json
	rm -rf __pycache__ pipeline/__pycache__

# Platform only — uses an Ollama running on the host (if any) for base-model chat.
docker-up:
	docker compose up -d --build

# Platform + a bundled Ollama container (fully self-contained). A base model
# (qwen2.5:0.5b) is auto-pulled at deploy time; override with OLLAMA_PULL_MODELS, e.g.
#   OLLAMA_PULL_MODELS="qwen2.5:0.5b qwen2.5:7b" make docker-up-ollama
docker-up-ollama:
	OLLAMA_HOST=http://ollama:11434 docker compose --profile ollama up -d --build

docker-down:
	docker compose --profile ollama down

lint:
	$(VENV)/pip install ruff
	$(VENV)/bin/ruff check pipeline/ ui/

test:
	$(VENV)/python -c "import mlx.core as mx; assert mx.metal.is_available(), 'Metal required'"
	@echo "✓ MLX Metal available"
