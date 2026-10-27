# stereo-depth

Disparity, depth and 3D point clouds from a stereo image pair or a live stereo camera.

| Path | What |
| --- | --- |
| `backend/` | FastAPI + PyTorch API and depth engines (`uv`) |
| `frontend/` | React + TypeScript + Vite web app (`pnpm`) |
| `cuda/` | Custom CUDA SGM kernels |
| `notebooks/` | Colab exploration only |

```sh
make setup   # install backend and frontend deps, pre-commit hooks
make lint
make test
make dev     # api on :7860, web on :8080
```

Branching is trunk-based: short-lived `feature/*`, `fix/*`, `chore/*` branches, squash-merged into a protected `main`.
