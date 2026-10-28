import cv2
import numpy as np
import numpy.typing as npt

from stereo_depth.engines.base import Image


def colorize(values: npt.NDArray[np.float32], vmax: float | None = None) -> Image:
    """Map a float map (disparity or depth) to an RGB turbo heatmap. NaN pixels are black."""
    valid = np.isfinite(values)
    if not valid.any():
        return np.zeros((*values.shape, 3), dtype=np.uint8)
    lo = float(np.min(values[valid]))
    hi = float(vmax) if vmax is not None else float(np.max(values[valid]))
    norm = np.zeros(values.shape, dtype=np.float32)
    if hi > lo:
        norm[valid] = np.clip((values[valid] - lo) / (hi - lo), 0.0, 1.0)
    gray = (norm * 255).astype(np.uint8)
    bgr = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    rgb = np.asarray(bgr[..., ::-1], dtype=np.uint8)
    rgb[~valid] = 0
    return np.ascontiguousarray(rgb)
