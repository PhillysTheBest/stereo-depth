"""Every engine the API can serve, built once per process."""

import contextlib
import importlib
from functools import cache

from stereo_depth.config import Settings
from stereo_depth.engines.base import DepthEngine
from stereo_depth.engines.raft_stereo import RAFTStereoEngine

LABELS = {
    "sgbm": "OpenCV SGBM (classical)",
    "raft_stereo": "RAFT-Stereo (pretrained network)",
    "cuda_sgm": "CUDA SGM (custom kernel)",
}

# SGBM (Sprint 1) and CUDA SGM (Sprint 3) live in their own modules; skip whichever is absent.
_OPTIONAL = [
    ("stereo_depth.engines.sgbm", "SGBMEngine"),
    ("stereo_depth.engines.cuda_sgm", "CudaSGMEngine"),
]


def _build(settings: Settings) -> dict[str, DepthEngine]:
    engines: dict[str, DepthEngine] = {}
    for module, cls in _OPTIONAL:
        with contextlib.suppress(ImportError):
            engine: DepthEngine = getattr(importlib.import_module(module), cls)()
            engines[engine.name] = engine
    raft = RAFTStereoEngine(settings)
    engines[raft.name] = raft
    order = list(LABELS)
    return dict(
        sorted(engines.items(), key=lambda kv: order.index(kv[0]) if kv[0] in order else 99)
    )


@cache
def get_engines() -> dict[str, DepthEngine]:
    return _build(Settings())
