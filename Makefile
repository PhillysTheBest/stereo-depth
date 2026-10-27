.PHONY: setup lint test dev bench cuda

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
