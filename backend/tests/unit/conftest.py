import numpy as np
import pytest

from stereo_depth.engines.base import Image

SHIFT = 8


@pytest.fixture
def shifted_pair() -> tuple[Image, Image]:
    """Random texture pair with constant disparity SHIFT: left[x] == right[x - SHIFT]."""
    rng = np.random.default_rng(0)
    gray = rng.integers(0, 256, size=(96, 256 + SHIFT), dtype=np.uint8)
    right = np.stack([gray[:, SHIFT:]] * 3, axis=-1)
    left = np.stack([gray[:, :-SHIFT]] * 3, axis=-1)
    return np.ascontiguousarray(left), np.ascontiguousarray(right)
