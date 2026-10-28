.PHONY: setup lint test dev bench data weights openapi cuda

setup:
	cd backend && uv sync --extra cpu
	cd frontend && pnpm install
	pre-commit install

lint:
	cd backend && uv run ruff check && uv run ruff format --check && uv run mypy
	cd frontend && pnpm lint && pnpm typecheck

test:
	cd backend && uv run --extra cpu pytest -m "not gpu"
	cd frontend && pnpm test

dev:
	docker compose up --build

bench:
	cd backend && uv run stereo-bench --dataset $(or $(DATASET),kitti2015) --root ../data/$(or $(DATASET),kitti2015) $(BENCH_ARGS)

cuda:
	$(MAKE) -C cuda

data:
	bash scripts/download_kitti.sh data/kitti2015
	bash scripts/download_middlebury.sh data/middlebury Q

weights:
	python scripts/fetch_weights.py

# Regenerate the OpenAPI schema and the frontend's typed client after changing the API.
openapi:
	cd backend && uv run python -m stereo_depth.api.openapi
	cd frontend && pnpm gen:api
