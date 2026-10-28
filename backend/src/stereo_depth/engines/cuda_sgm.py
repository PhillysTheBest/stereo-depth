from importlib import import_module
from time import perf_counter
from types import ModuleType

from stereo_depth.engines.base import DepthEngine, Disparity, Image
from stereo_depth.sgm import SgmParams, to_gray

# Built from cuda/ with `pip install --no-build-isolation ./cuda`; not on PyPI.
EXTENSION = "stereo_depth_cuda"


def _load() -> tuple[ModuleType, ModuleType] | None:
    try:
        return import_module("torch"), import_module(EXTENSION)
    except ImportError:
        return None


class CudaSgmEngine(DepthEngine):
    """SGM on the GPU via the `stereo_depth_cuda` PyTorch extension. Matches `SgmCpuEngine`."""

    name = "cuda-sgm"
    requires_gpu = True

    def __init__(self, params: SgmParams | None = None) -> None:
        self.params = params or SgmParams()

    def is_available(self) -> bool:
        modules = _load()
        return modules is not None and bool(modules[0].cuda.is_available())

    def predict(self, left: Image, right: Image) -> Disparity:
        modules = _load()
        if modules is None or not modules[0].cuda.is_available():
            raise RuntimeError(f"{self.name} needs a CUDA GPU, torch and the {EXTENSION} extension")
        torch, ext = modules
        start = perf_counter()
        gpu_left = torch.from_numpy(to_gray(left)).cuda()
        gpu_right = torch.from_numpy(to_gray(right)).cuda()
        p = self.params
        out = ext.disparity(gpu_left, gpu_right, p.max_disp, p.p1, p.p2)
        values = out.cpu().numpy()
        return Disparity(values, runtime_ms=(perf_counter() - start) * 1000)
