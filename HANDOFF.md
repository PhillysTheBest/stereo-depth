# Handoff

Status as of 2026-10-27. Branch `chore/ci-workflows` is pushed as PR #1 with every CI check green. It is waiting for a squash-merge.

## Current state

Done and checked locally:

- Repo layout: `backend/`, `frontend/`, `cuda/`, `notebooks/`, `scripts/`, `data/`, `docs/decisions/`
- `notebooks/oracle.ipynb` holds the Colab CUDA 12.8 test on a T4 (merged from `main`, outputs stripped)
- Backend: uv project, FastAPI `/v1/health`, `DepthEngine` and `CameraSource` stubs, settings, 2 passing tests and 1 test marked `gpu`. ruff, ruff format, mypy (strict) and pytest all pass
- Frontend: Vite + React 18 + TS scaffold, 1 passing vitest test. eslint, tsc and the Vite build pass. TypeScript is pinned to 5.8 because typescript-eslint doesn't support TS 7
- Lockfiles committed: `backend/uv.lock`, `frontend/pnpm-lock.yaml`
- Root: `Makefile`, `docker-compose.yml`, `.gitignore`, `.pre-commit-config.yaml`, README
- Git identity for this repo set locally to `PhillysTheBest <151466366+PhillysTheBest@users.noreply.github.com>`, with no co-author lines

Checked in CI (PR #1): `backend`, `frontend`, `cuda-build`, `notebooks`, `docker-build (api)` and `docker-build (web)` all pass.

Repo settings applied: squash merge only, auto-delete head branches, and ruleset `protect-main` (PR with 0 approvals, the five required checks, linear history, no force pushes or deletion).

Not yet run: `cd.yml`, `release.yml`, `gpu-tests.yml`, `backend/Dockerfile.gpu`, and Dependabot (it starts once `dependabot.yml` is on `main`).

## To complete

### Sprint 0: Foundations (finish first)

- [x] Push `chore/ci-workflows` and open the PR (`gh pr create`)
- [x] Fix whatever fails on the first CI run (only a merge conflict with `main` over `oracle.ipynb`; every check passed)
- [ ] Check the Docker images build and `/v1/health` responds (`make dev`). CI builds both images, but Docker Desktop's WSL integration is off, so `make dev` hasn't run
- [ ] Squash-merge the PR once every check is green
- [ ] Confirm `cd.yml` pushed `stereo-depth-api`, `stereo-depth-api-gpu` and `stereo-depth-web` to GHCR
- [x] Protect `main` (Settings, then Rules): require a PR with 0 approvals; require the checks `backend`, `frontend`, `cuda-build` and both `docker-build` matrix jobs; require linear history; allow squash merge only; block force pushes and deletion
- [x] Turn on auto-delete of head branches
- [ ] Check that Dependabot is active for uv, npm, docker and github-actions
- [ ] Link the Vercel project to the repo (root directory `frontend/`) and check each PR gets a preview
- [ ] Make GHCR packages public if wanted (they're private by default)
- [ ] Prove it: a trivial PR goes all green, merging it pushes images, and a direct push to `main` is rejected
- [ ] Optional: install pre-commit locally (`make setup`) and run `pre-commit run --all-files`
- [x] Optional: put real CUDA test code in `notebooks/oracle.ipynb`, or delete it

### Sprint 1: Oracle and evaluation (tag v0.1.0)

Backend
- [ ] Add `opencv-python-headless`, Pillow and imageio
- [ ] `engines/sgbm.py`: StereoSGBM engine implementing `DepthEngine`
- [ ] `core/`: image I/O, downscale to 1280 px max, disparity to depth, PLY export (`plyfile`)
- [ ] `scripts/download_kitti.sh`, `scripts/download_middlebury.sh`
- [ ] `eval/`: KITTI 2015 and Middlebury loaders, EPE, D1-all and bad-2.0 metrics, runtime per pair
- [ ] Benchmark CLI and `make bench`
- [ ] Log SGBM scores on the KITTI 2015 train split (W&B)

Frontend
- [ ] Upload page for a left and right image
- [ ] Mock API client
- [ ] Result layout

Release
- [ ] Tag `v0.1.0` and check `release.yml` retags the images and creates a GitHub Release

### Sprint 2: Model and API (tag v0.2.0)

Backend
- [ ] RAFT-Stereo engine (check its licence) and `scripts/fetch_weights.py` pulling weights from Hugging Face
- [ ] Run RAFT-Stereo on CPU at reduced size
- [ ] Add CPU-only PyTorch wheels to CI
- [ ] `POST /v1/depth`: multipart left/right, engine, optional `focal_px` and `baseline_m`; returns disparity and depth PNGs, depth stats, PLY link and runtime
- [ ] `GET /v1/engines`, which hides GPU engines on CPU hosts
- [ ] Extend `/v1/health` with model-loaded status
- [ ] Return relative depth when there's no calibration, and don't store images
- [ ] Publish the OpenAPI schema
- [ ] Check `Dockerfile.gpu` works with the `[gpu]` extra

Frontend
- [ ] Typed client generated from OpenAPI into `src/api/`
- [ ] ModelPicker for choosing the engine
- [ ] Disparity and depth heatmaps with a colour scale
- [ ] Loading and error states

Done when
- [ ] A browser upload returns depth from both engines, and a benchmark table compares them
- [ ] Tag `v0.2.0`

### Sprint 3: CUDA and 3D (tag v0.3.0)

Backend and CUDA
- [ ] CPU SGM reference in NumPy and Numba
- [ ] CUDA kernels: cost volume, 4-path aggregation, winner-take-all (8 paths and sub-pixel come later)
- [ ] `cuda/bindings.cpp` and `cuda/setup.py` building the `stereo_depth_cuda` PyTorch extension
- [ ] `engines/cuda_sgm.py`
- [ ] GPU tests and a speed benchmark in Colab
- [ ] Notebooks install the package with `pip install git+...` and log to W&B

Frontend
- [ ] PointCloud viewer (react-three-fiber)
- [ ] Hover a pixel to read its depth
- [ ] PNG and PLY downloads
- [ ] Benchmarks page

Done when
- [ ] CUDA output is within 1% bad-px of the CPU reference on Middlebury and faster on a T4
- [ ] Camera ordered
- [ ] Tag `v0.3.0`

### Sprint 4: Ship v1.0 (tag v1.0.0)

Backend
- [ ] Deploy the CPU API to Hugging Face Spaces, with `HF_TOKEN` as a repo secret
- [ ] Rate limit and upload size cap
- [ ] Structured logging
- [ ] Staging deploy in `cd.yml` and production deploy in `release.yml`

Frontend
- [ ] Production deploy on Vercel, with the API URL set as an env var
- [ ] Sample image pairs (our own, not KITTI)
- [ ] Tailwind and mobile layout
- [ ] Playwright upload-to-depth test in `e2e/`

Docs
- [ ] README with a demo GIF and architecture notes
- [ ] `docs/architecture.md`, `docs/api.md` and ADRs in `docs/decisions/`

Release
- [ ] Public URL works end to end
- [ ] Tag `v1.0.0`

### Sprint 5: Stereo camera (tag v1.1.0)

Backend
- [ ] `SimulatedCamera` replaying KITTI sequences
- [ ] `WS /v1/stream` at 320–640 px, using SGBM for live frames
- [ ] `POST /v1/calibrate` with checkerboard photos, returning intrinsics, baseline and rectification maps
- [ ] Adapter for the purchased camera

Frontend
- [ ] Live page showing the camera feed and depth side by side (getUserMedia, split into left and right halves)
- [ ] Calibration wizard
- [ ] FPS and latency readout

Done when
- [ ] Live depth runs end to end on the simulated camera in CI
- [ ] It runs on the real camera once it arrives
- [ ] Tag `v1.1.0`

### Stretch (after v1.0)

- [ ] CUDA engine live on a GPU laptop or a Jetson board
- [ ] Fine-tuning on your own data
- [ ] A live GPU host
- [ ] ONNX and TensorRT export

### Open decisions

- [ ] Which stereo camera (the plan assumes a USB dual-lens UVC webcam)
- [ ] Your laptop OS (not blocking)
- [ ] Real distances vs relative depth by default
