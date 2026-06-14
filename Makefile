.PHONY: install run train export clean docker-up docker-down

VENV = .venv/bin

install: $(VENV)/python
	$(VENV)/pip install -r requirements.txt

$(VENV)/python:
	python3 -m venv .venv
	$(VENV)/pip install --upgrade pip

run: install
	$(VENV)/python ui/app.py

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

docker-up:
	docker compose up -d --build

docker-down:
	docker compose down

lint:
	$(VENV)/pip install ruff
	$(VENV)/bin/ruff check pipeline/ ui/

test:
	$(VENV)/python -c "import mlx.core as mx; assert mx.metal.is_available(), 'Metal required'"
	@echo "✓ MLX Metal available"
