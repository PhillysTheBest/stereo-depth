import numpy as np
import pytest

from stereo_depth.engines import SGBMEngine
from stereo_depth.engines.base import Image
from tests.unit.conftest import SHIFT


def test_sgbm_recovers_constant_shift(shifted_pair: tuple[Image, Image]) -> None:
    left, right = shifted_pair
    out = SGBMEngine(num_disparities=32).predict(left, right)
    assert out.values.shape == left.shape[:2]
    assert out.values.dtype == np.float32
    interior = out.values[10:-10, 48:-10]
    assert np.nanmedian(interior) == pytest.approx(SHIFT, abs=0.25)
    assert np.isfinite(interior).mean() > 0.9
    assert out.runtime_ms >= 0


def test_sgbm_marks_unmatched_border_invalid(shifted_pair: tuple[Image, Image]) -> None:
    left, right = shifted_pair
    out = SGBMEngine(num_disparities=32).predict(left, right)
    # Columns left of numDisparities cannot be matched, so they must be NaN, not 0.
    assert not np.any(out.values == 0)
    assert np.isnan(out.values[:, :16]).all()


def test_sgbm_rejects_bad_params() -> None:
    with pytest.raises(ValueError):
        SGBMEngine(num_disparities=30)
    with pytest.raises(ValueError):
        SGBMEngine(block_size=4)


def test_sgbm_rejects_mismatched_pair() -> None:
    a = np.zeros((10, 20, 3), dtype=np.uint8)
    b = np.zeros((10, 22, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        SGBMEngine().predict(a, b)
