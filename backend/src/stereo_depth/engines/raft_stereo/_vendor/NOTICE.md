# Vendored RAFT-Stereo

Source: https://github.com/princeton-vl/RAFT-Stereo, `core/` on the `main` branch, fetched 2026-10-28.
Licence: MIT, copyright 2021 Princeton Vision & Learning Lab (see `LICENSE` in this folder).
Paper: Lipson, Teed and Deng, "RAFT-Stereo: Multilevel Recurrent Field Transforms for Stereo Matching", 3DV 2021.

Only inference code is kept. Local changes, so it runs on current PyTorch without extra deps:

- Imports are relative (`core.x` becomes `.x`); `core/utils/utils.py` is `utils.py`.
- `opt_einsum` import removed (it was unused).
- `forward_interpolate` and its `scipy` import removed (training only).
- `torch.meshgrid` calls pass `indexing="ij"`, the behaviour the code already relied on.
- `torch.cuda.amp.autocast` replaced by `torch.autocast("cuda")`, falling back to a no-op on CPU.
- Bare `except:` around the optional CUDA samplers narrowed to `except ImportError:`.

Lint and type checks skip this folder; the wrapper in `../engine.py` is checked.
