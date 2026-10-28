import io

import numpy as np
import pytest
from PIL import Image as PILImage

from stereo_depth.api import app as app_module
from stereo_depth.engines import DepthEngine, Disparity
from stereo_depth.engines.base import Image


class ConstantEngine(DepthEngine):
    """Returns the same disparity everywhere, with the left column invalid."""

    name = "constant"

    def __init__(self, value: float = 8.0) -> None:
        self.value = value
        self.calls = 0

    def predict(self, left: Image, right: Image) -> Disparity:
        self.calls += 1
        values = np.full(left.shape[:2], self.value, dtype=np.float32)
        values[:, 0] = np.nan
        return Disparity(values=values, runtime_ms=1.5)


class GpuOnlyEngine(ConstantEngine):
    name = "gpu_only"
    requires_gpu = True


class MissingWeightsEngine(ConstantEngine):
    name = "missing"

    def is_available(self) -> bool:
        return False


@pytest.fixture
def fake_engines(monkeypatch: pytest.MonkeyPatch) -> dict[str, DepthEngine]:
    engines: dict[str, DepthEngine] = {
        e.name: e for e in (ConstantEngine(), GpuOnlyEngine(), MissingWeightsEngine())
    }
    monkeypatch.setattr(app_module, "get_engines", lambda: engines)
    monkeypatch.setattr(app_module, "_gpu_available", lambda: False)
    return engines


def png_bytes(width: int, height: int, seed: int = 0) -> bytes:
    rng = np.random.default_rng(seed)
    pixels = rng.integers(0, 255, (height, width, 3), dtype=np.uint8)
    buf = io.BytesIO()
    PILImage.fromarray(pixels).save(buf, format="PNG")
    return buf.getvalue()
