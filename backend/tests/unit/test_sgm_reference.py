import itertools

import numpy as np
import numpy.typing as npt
import pytest

from stereo_depth.sgm import (
    INVALID_COST,
    SgmParams,
    aggregate,
    census,
    cost_volume,
    sgm_disparity,
    to_gray,
)
from tests.stereo_pairs import shifted_noise_pair


def naive_aggregate(cost: npt.NDArray[np.uint8], p1: int, p2: int) -> npt.NDArray[np.int64]:
    """Textbook per-pixel SGM recurrence, to check the vectorised version against."""
    h, w, d = cost.shape
    total = np.zeros((h, w, d), dtype=np.int64)
    for dy, dx in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        ys = range(h) if dy >= 0 else range(h - 1, -1, -1)
        xs = range(w) if dx >= 0 else range(w - 1, -1, -1)
        path = np.zeros((h, w, d), dtype=np.int64)
        for y, x in itertools.product(ys, xs):
            py, px = y - dy, x - dx
            if not (0 <= py < h and 0 <= px < w):
                path[y, x] = cost[y, x]
                continue
            prev = path[py, px]
            m = prev.min()
            for k in range(d):
                best = min(prev[k], m + p2)
                if k > 0:
                    best = min(best, prev[k - 1] + p1)
                if k < d - 1:
                    best = min(best, prev[k + 1] + p1)
                path[y, x, k] = int(cost[y, x, k]) + best - m
        total += path
    return total


def test_to_gray_matches_bt601() -> None:
    rgb = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 255]]], dtype=np.uint8)
    assert to_gray(rgb).tolist() == [[77, 149, 29, 255]]


def test_census_of_flat_image_is_zero() -> None:
    assert not census(np.full((10, 12), 7, dtype=np.uint8)).any()


def test_census_sets_one_bit_per_darker_neighbour() -> None:
    img = np.zeros((9, 11), dtype=np.uint8)
    img[4, 5] = 255
    bits = census(img)
    assert int(bits[4, 5]).bit_count() == 62
    # Every other pixel has the bright centre as its only non-darker neighbour.
    assert int(bits[0, 0]) == 0


def test_cost_volume_marks_disparities_past_left_edge() -> None:
    c = census(np.random.default_rng(1).integers(0, 256, (4, 8), dtype=np.uint8))
    cost = cost_volume(c, c, 32)
    assert (cost[:, 0, 1:] == INVALID_COST).all()
    assert (cost[:, :, 0] == 0).all()
    assert (cost[:, 7, 8:] == INVALID_COST).all()


def test_aggregate_matches_naive_recurrence() -> None:
    cost = np.random.default_rng(2).integers(0, INVALID_COST + 1, (5, 7, 6), dtype=np.uint8)
    assert np.array_equal(aggregate(cost, 3, 20), naive_aggregate(cost, 3, 20))


def test_recovers_constant_shift() -> None:
    left, right = shifted_noise_pair(48, 96, shift=13)
    disp = sgm_disparity(left, right, SgmParams(max_disp=32))
    interior = disp[:, 13 + 4 :]
    assert (interior == 13).mean() > 0.98


@pytest.mark.parametrize("kwargs", [{"max_disp": 48}, {"max_disp": 0}, {"p1": 50, "p2": 40}])
def test_rejects_bad_params(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        SgmParams(**kwargs)
