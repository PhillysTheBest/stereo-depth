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

## Benchmarks

```sh
make data                          # KITTI 2015 (~1.7 GB) and Middlebury v3 quarter-res into data/
make bench                         # KITTI 2015 training split
make bench DATASET=middlebury
make bench BENCH_ARGS="--limit 20 --out results.json --wandb stereo-depth"
```

Disparity holes are filled by KITTI devkit background interpolation before scoring; `density` is the share of ground-truth pixels with a real prediction. D1-all is the KITTI outlier rate (error > 3 px and > 5%), bad-2.0 the share with error > 2 px.

| Engine | Dataset | EPE (px) | D1-all | bad-2.0 | Density | Runtime (CPU) |
| --- | --- | --- | --- | --- | --- | --- |
| SGBM (OpenCV, 128 disp, 5 px block) | KITTI 2015 train, 200 pairs | 1.83 | 9.97% | 15.08% | 83.9% | 48 ms/pair |
