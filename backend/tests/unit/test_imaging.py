import numpy as np
import pytest

from stereo_depth.api import imaging


def test_relative_depth_is_median_normalised() -> None:
    disp = np.array([[1.0, 2.0, 4.0, np.nan, 0.0]], dtype=np.float32)
    rel = imaging.relative_depth(disp)
    assert np.nanmedian(rel) == pytest.approx(1.0)
    assert rel[0, 0] > rel[0, 1] > rel[0, 2]  # larger disparity is nearer
    assert np.isnan(rel[0, 3:]).all()


def test_robust_range_ignores_nan_and_flat_maps() -> None:
    assert imaging.robust_range(np.full((2, 2), np.nan, np.float32)) == (0.0, 1.0)
    assert imaging.robust_range(np.full((2, 2), 3.0, np.float32)) == (3.0, 4.0)


@pytest.mark.parametrize("reverse", [False, True])
def test_colorize_range_ends_and_nan(reverse: bool) -> None:
    values = np.array([[0.0, 1.0, np.nan]], dtype=np.float32)
    rgb = imaging.colorize_range(values, 0.15, 0.95, reverse=reverse)
    assert rgb.dtype == np.uint8 and rgb.shape == (1, 3, 3)
    low, high = (rgb[0, 1], rgb[0, 0]) if reverse else (rgb[0, 0], rgb[0, 1])
    assert int(low[2]) > int(low[0])  # Turbo's low end is blue
    assert int(high[0]) > int(high[2])  # and its high end red
    assert (rgb[0, 2] == 0).all()
