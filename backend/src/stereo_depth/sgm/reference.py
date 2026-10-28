"""CPU reference for semi-global matching (SGM).

This is the ground truth the CUDA kernels in `cuda/sgm/sgm.cu` are checked against, so every
step uses integer maths with the same rounding, border handling and tie-breaking:

1. Census transform over a 9x7 window, borders clamped to the nearest pixel.
2. Matching cost is the Hamming distance between left census at x and right census at x - d.
   Disparities that fall off the left edge get `INVALID_COST`.
3. Costs are aggregated along 4 paths (left, right, up, down) with penalties P1 and P2.
4. Winner-take-all picks the lowest summed cost, taking the smallest disparity on ties.

Each path is vectorised across scanlines and disparities, so the Python loop runs once per
pixel along the path rather than once per pixel.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

CENSUS_W = 9
CENSUS_H = 7
# One more than the largest Hamming distance a 9x7 census can produce (62 bits).
INVALID_COST = 64

Gray = npt.NDArray[np.uint8]
Census = npt.NDArray[np.uint64]
CostVolume = npt.NDArray[np.uint8]
CostSum = npt.NDArray[np.uint16]


@dataclass(frozen=True)
class SgmParams:
    """`max_disp` must be a multiple of 32 (one CUDA warp per 32 disparities), up to 1024."""

    max_disp: int = 128
    p1: int = 10
    p2: int = 120

    def __post_init__(self) -> None:
        if self.max_disp <= 0 or self.max_disp % 32 or self.max_disp > 1024:
            raise ValueError(f"max_disp must be a multiple of 32 in 32..1024, got {self.max_disp}")
        if not 0 < self.p1 < self.p2 <= 255 - INVALID_COST:
            raise ValueError(f"need 0 < p1 < p2 <= {255 - INVALID_COST}")


def to_gray(image: npt.NDArray[np.uint8]) -> Gray:
    """BT.601 luma in fixed point, matching OpenCV's RGB2GRAY to within 1 level."""
    if image.ndim == 2:
        return np.ascontiguousarray(image, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError(f"expected HxW or HxWx3/4 image, got shape {image.shape}")
    rgb = image[..., :3].astype(np.uint32)
    luma = (77 * rgb[..., 0] + 150 * rgb[..., 1] + 29 * rgb[..., 2] + 128) >> 8
    return luma.astype(np.uint8)


def census(gray: Gray) -> Census:
    """9x7 census: one bit per neighbour, set when the neighbour is darker than the centre."""
    h, w = gray.shape
    ry, rx = CENSUS_H // 2, CENSUS_W // 2
    padded = np.pad(gray, ((ry, ry), (rx, rx)), mode="edge")
    bits = np.zeros((h, w), dtype=np.uint64)
    one = np.uint64(1)
    for dy in range(-ry, ry + 1):
        for dx in range(-rx, rx + 1):
            if dy == 0 and dx == 0:
                continue
            neighbour = padded[ry + dy : ry + dy + h, rx + dx : rx + dx + w]
            bits = (bits << one) | (neighbour < gray).astype(np.uint64)
    return bits


def cost_volume(census_left: Census, census_right: Census, max_disp: int) -> CostVolume:
    """Hamming cost, shape (H, W, max_disp)."""
    h, w = census_left.shape
    cost = np.full((h, w, max_disp), INVALID_COST, dtype=np.uint8)
    for d in range(min(max_disp, w)):
        diff = census_left[:, d:] ^ census_right[:, : w - d]
        cost[:, d:, d] = np.bitwise_count(diff)
    return cost


def _scan(cost: CostVolume, total: CostSum, p1: int, p2: int) -> None:
    """Aggregate along axis 1 of (N, L, D) views and add the path cost into `total`."""
    n, length, d = cost.shape
    big = np.iinfo(np.int32).max // 2
    prev = cost[:, 0, :].astype(np.int32)
    total[:, 0, :] += prev.astype(np.uint16)
    shifted = np.empty((n, d), dtype=np.int32)
    for i in range(1, length):
        min_prev = prev.min(axis=1, keepdims=True)
        best = np.minimum(prev, min_prev + p2)
        shifted[:, 0] = big
        shifted[:, 1:] = prev[:, :-1] + p1
        np.minimum(best, shifted, out=best)
        shifted[:, -1] = big
        shifted[:, :-1] = prev[:, 1:] + p1
        np.minimum(best, shifted, out=best)
        prev = cost[:, i, :].astype(np.int32) + best - min_prev
        total[:, i, :] += prev.astype(np.uint16)


def aggregate(cost: CostVolume, p1: int, p2: int) -> CostSum:
    """Sum of the 4 path costs (left to right, right to left, top down, bottom up)."""
    total = np.zeros(cost.shape, dtype=np.uint16)
    _scan(cost, total, p1, p2)
    _scan(cost[:, ::-1], total[:, ::-1], p1, p2)
    vertical_cost, vertical_total = cost.transpose(1, 0, 2), total.transpose(1, 0, 2)
    _scan(vertical_cost, vertical_total, p1, p2)
    _scan(vertical_cost[:, ::-1], vertical_total[:, ::-1], p1, p2)
    return total


def winner_take_all(total: CostSum) -> npt.NDArray[np.float32]:
    # argmin returns the first minimum, so ties go to the smallest disparity.
    return total.argmin(axis=2).astype(np.float32)


def sgm_disparity(left: Gray, right: Gray, params: SgmParams) -> npt.NDArray[np.float32]:
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError(f"need two grey images of the same shape, got {left.shape} {right.shape}")
    cost = cost_volume(census(left), census(right), params.max_disp)
    return winner_take_all(aggregate(cost, params.p1, params.p2))
