from pathlib import Path

import imageio.v3 as iio
import numpy as np

from stereo_depth.engines.base import Image


def _to_rgb(img: np.ndarray) -> Image:
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if img.ndim != 3 or img.shape[2] not in (3, 4):
        raise ValueError(f"unsupported image shape {img.shape}")
    img = img[..., :3]
    if img.dtype == np.uint16:
        img = (img >> 8).astype(np.uint8)
    elif img.dtype != np.uint8:
        raise ValueError(f"unsupported image dtype {img.dtype}")
    return np.ascontiguousarray(img)


def decode_image(data: bytes) -> Image:
    """Decode PNG/JPEG/etc. bytes to an HxWx3 uint8 RGB array."""
    try:
        img = iio.imread(data)
    except Exception as exc:
        raise ValueError("could not decode image") from exc
    return _to_rgb(np.asarray(img))


def load_image(path: str | Path) -> Image:
    """Read an image file as HxWx3 uint8 RGB."""
    return decode_image(Path(path).read_bytes())


def encode_png(img: np.ndarray) -> bytes:
    return bytes(iio.imwrite("<bytes>", img, extension=".png"))
