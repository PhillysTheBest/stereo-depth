.PHONY: setup lint test dev bench cuda cuda-ext

setup:
	cd backend && uv sync
	cd frontend && pnpm install
	pre-commit install

lint:
	cd backend && uv run ruff check && uv run ruff format --check && uv run mypy
	cd frontend && pnpm lint && pnpm typecheck

test:
	cd backend && uv run pytest -m "not gpu"
	cd frontend && pnpm test

dev:
	docker compose up --build

bench:
	@echo "make bench arrives in Sprint 1"

cuda:
	$(MAKE) -C cuda

# Needs nvcc and a CUDA build of torch in the backend venv; run on a GPU machine or Colab.
cuda-ext:
	cd backend && uv pip install setuptools && uv pip install --no-build-isolation ../cuda
