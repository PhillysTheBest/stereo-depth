import time

import cv2
import numpy as np

from stereo_depth.engines.base import DepthEngine, Disparity, Image


class SGBMEngine(DepthEngine):
    """OpenCV semi-global block matching. The CPU oracle every other engine is checked against."""

    name = "sgbm"

    def __init__(self, num_disparities: int = 128, block_size: int = 5) -> None:
        if num_disparities <= 0 or num_disparities % 16:
            raise ValueError("num_disparities must be a positive multiple of 16")
        if block_size < 1 or block_size % 2 == 0:
            raise ValueError("block_size must be odd")
        self.num_disparities = num_disparities
        self.block_size = block_size
        channels = 3
        self._matcher = cv2.StereoSGBM.create(
            minDisparity=0,
            numDisparities=num_disparities,
            blockSize=block_size,
            P1=8 * channels * block_size**2,
            P2=32 * channels * block_size**2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=2,
            preFilterCap=63,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )

    def predict(self, left: Image, right: Image) -> Disparity:
        if left.shape != right.shape:
            raise ValueError("left and right images must have the same shape")
        start = time.perf_counter()
        raw = self._matcher.compute(left, right)
        runtime_ms = (time.perf_counter() - start) * 1000
        # OpenCV returns fixed-point disparity with 4 fractional bits; negatives are invalid.
        disp = np.asarray(raw, dtype=np.float32) / 16.0
        disp[disp <= 0] = np.nan
        return Disparity(values=disp, runtime_ms=runtime_ms)
