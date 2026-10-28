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

## CUDA SGM

`cuda/` holds a semi-global matching kernel (census cost, 4-path aggregation, winner-take-all)
and its PyTorch bindings. It must match the NumPy reference in `backend/src/stereo_depth/sgm/`.

```sh
make cuda                     # compile the kernels and a self-test (nvcc, no GPU needed)
make -C cuda test             # run the self-test (GPU)
make cuda-ext                 # build the stereo_depth_cuda extension into the backend venv
```

CI only compiles. Kernels run in Colab via `notebooks/cuda_sgm.ipynb`, which runs `pytest -m gpu`
and benchmarks against the CPU reference on Middlebury Cones.

Branching is trunk-based: short-lived `feature/*`, `fix/*`, `chore/*` branches, squash-merged into a protected `main`.
