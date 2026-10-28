import numpy as np
import numpy.typing as npt


def disparity_to_depth(
    disp: npt.NDArray[np.float32], focal_px: float, baseline_m: float
) -> npt.NDArray[np.float32]:
    """depth = focal * baseline / disparity, in metres. NaN where disparity is <= 0 or NaN."""
    if focal_px <= 0 or baseline_m <= 0:
        raise ValueError("focal_px and baseline_m must be positive")
    valid = np.isfinite(disp) & (disp > 0)
    depth = np.full(disp.shape, np.nan, dtype=np.float32)
    depth[valid] = np.float32(focal_px * baseline_m) / disp[valid]
    return depth
