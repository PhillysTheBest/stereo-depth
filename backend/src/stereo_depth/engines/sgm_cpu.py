from time import perf_counter

from stereo_depth.engines.base import DepthEngine, Disparity, Image
from stereo_depth.sgm import SgmParams, sgm_disparity, to_gray


class SgmCpuEngine(DepthEngine):
    """NumPy SGM. Slow, but it's the reference the CUDA engine must match."""

    name = "sgm-cpu"

    def __init__(self, params: SgmParams | None = None) -> None:
        self.params = params or SgmParams()

    def predict(self, left: Image, right: Image) -> Disparity:
        start = perf_counter()
        values = sgm_disparity(to_gray(left), to_gray(right), self.params)
        return Disparity(values, runtime_ms=(perf_counter() - start) * 1000)
