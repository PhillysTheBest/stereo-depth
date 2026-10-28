"""Heatmap and relative-depth helpers for the depth endpoint (the rest is in ``core``)."""

import cv2
import numpy as np
import numpy.typing as npt

from stereo_depth.engines.base import Image

FloatMap = npt.NDArray[np.float32]


def relative_depth(disp: FloatMap) -> FloatMap:
    """Inverse disparity scaled so the median valid pixel is 1.0; unitless, NaN where invalid."""
    valid = np.isfinite(disp) & (disp > 0)
    rel = np.full(disp.shape, np.nan, dtype=np.float32)
    rel[valid] = 1.0 / disp[valid]
    if valid.any():
        rel /= np.float32(np.median(rel[valid]))
    return rel


def robust_range(values: FloatMap) -> tuple[float, float]:
    """2nd to 98th percentile of the finite values, so a few outliers don't wash out the map."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(finite, [2, 98])
    return float(lo), float(hi if hi > lo else lo + 1.0)


def colorize_range(values: FloatMap, vmin: float, vmax: float, reverse: bool = False) -> Image:
    """Turbo heatmap over a fixed [vmin, vmax], so the UI legend matches; NaN pixels are black.

    ``reverse`` puts vmin at the red end (used for depth, so near is red).
    """
    valid = np.isfinite(values)
    t = np.zeros(values.shape, dtype=np.float32)
    t[valid] = np.clip((values[valid] - vmin) / (vmax - vmin), 0.0, 1.0)
    if reverse:
        t = 1.0 - t
    bgr = cv2.applyColorMap((t * 255).round().astype(np.uint8), cv2.COLORMAP_TURBO)
    rgb = np.ascontiguousarray(bgr[..., ::-1], dtype=np.uint8)
    rgb[~valid] = 0
    return rgb
