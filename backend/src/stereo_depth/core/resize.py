import cv2
import numpy as np
import numpy.typing as npt

from stereo_depth.engines.base import Image


def downscale(img: Image, max_px: int) -> tuple[Image, float]:
    """Shrink so the longer side is at most max_px. Returns the image and the scale (<= 1.0).

    Disparity computed on the downscaled pair is in downscaled pixels; divide by scale for
    full-resolution pixels.
    """
    h, w = img.shape[:2]
    scale = min(1.0, max_px / max(h, w))
    if scale == 1.0:
        return img, 1.0
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    out = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    return np.asarray(out, dtype=np.uint8), scale


def upscale_disparity(
    disp: npt.NDArray[np.float32], shape: tuple[int, int], scale: float
) -> npt.NDArray[np.float32]:
    """Resize a disparity map back to (h, w) and convert it to full-resolution pixels."""
    if scale == 1.0 and disp.shape == shape:
        return disp
    out = cv2.resize(disp, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return (np.asarray(out, dtype=np.float32) / np.float32(scale)).astype(np.float32)
