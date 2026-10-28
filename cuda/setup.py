"""Builds the `stereo_depth_cuda` PyTorch extension.

    pip install --no-build-isolation ./cuda
    pip install --no-build-isolation "git+https://github.com/PhillysTheBest/stereo-depth#subdirectory=cuda"

--no-build-isolation makes it compile against the torch already installed (Colab's, for example).
Set TORCH_CUDA_ARCH_LIST to pick GPUs; it defaults to the GPU torch can see.
"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="stereo-depth-cuda",
    version="0.3.0",
    description="Semi-global matching stereo kernels for stereo-depth",
    ext_modules=[
        CUDAExtension(
            name="stereo_depth_cuda",
            sources=["bindings.cpp", "sgm/sgm.cu"],
            extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3", "-lineinfo"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
