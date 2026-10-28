"""Checks the CUDA kernels against the NumPy reference. Run in Colab: notebooks/cuda_sgm.ipynb."""

import numpy as np
import pytest

from stereo_depth.engines import CudaSGMEngine, SGMCpuEngine
from stereo_depth.sgm import SgmParams, aggregate, census, cost_volume
from tests.stereo_pairs import shifted_noise_pair

pytestmark = pytest.mark.gpu
torch = pytest.importorskip("torch")
ext = pytest.importorskip("stereo_depth_cuda")


def to_gpu(a: np.ndarray) -> "torch.Tensor":
    return torch.from_numpy(a).cuda()


def test_census_matches_reference() -> None:
    left, _ = shifted_noise_pair(37, 53, shift=0)
    got = ext.census(to_gpu(left)).cpu().numpy().view(np.uint64)
    assert np.array_equal(got, census(left))


def test_cost_sum_matches_reference() -> None:
    left, right = shifted_noise_pair(29, 71, shift=9, seed=3)
    params = SgmParams(max_disp=64)
    expected = aggregate(cost_volume(census(left), census(right), 64), params.p1, params.p2)
    got = ext.cost_sum(to_gpu(left), to_gpu(right), 64, params.p1, params.p2).cpu().numpy()
    assert np.array_equal(got.astype(np.uint16), expected)


@pytest.mark.parametrize("max_disp", [32, 64, 128])
def test_disparity_within_one_percent_of_reference(max_disp: int) -> None:
    left, right = shifted_noise_pair(120, 200, shift=17, seed=max_disp)
    params = SgmParams(max_disp=max_disp)
    cpu = SGMCpuEngine(params).predict(left, right).values
    gpu = CudaSGMEngine(params).predict(left, right).values
    assert (np.abs(cpu - gpu) > 1).mean() < 0.01


def test_rejects_cpu_tensors() -> None:
    img = torch.zeros((8, 8), dtype=torch.uint8)
    with pytest.raises(RuntimeError):
        ext.disparity(img, img, 32, 10, 120)
