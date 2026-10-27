import numpy as np

from stereo_depth.engines import DepthEngine, Disparity
from stereo_depth.engines.base import Image


class ZeroEngine(DepthEngine):
    name = "zero"

    def predict(self, left: Image, right: Image) -> Disparity:
        return Disparity(np.zeros(left.shape[:2], dtype=np.float32), runtime_ms=0.0)


def test_engine_returns_left_aligned_disparity() -> None:
    left = np.zeros((4, 6, 3), dtype=np.uint8)
    out = ZeroEngine().predict(left, left)
    assert out.values.shape == (4, 6)
