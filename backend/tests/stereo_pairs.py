import numpy as np
import numpy.typing as npt

Pair = tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]


def shifted_noise_pair(h: int, w: int, shift: int, seed: int = 0) -> Pair:
    """Random texture where every left pixel x matches right pixel x - shift."""
    base = np.random.default_rng(seed).integers(0, 256, (h, w + shift), dtype=np.uint8)
    return base[:, :w].copy(), base[:, shift : shift + w].copy()
