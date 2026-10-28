import numpy as np
import pytest

from stereo_depth.engines import CudaSGMEngine, SGMCpuEngine, cuda_sgm
from stereo_depth.sgm import SgmParams
from tests.stereo_pairs import shifted_noise_pair


def test_cpu_engine_accepts_rgb() -> None:
    left, right = shifted_noise_pair(24, 64, shift=5)
    rgb = np.stack([left] * 3, axis=-1), np.stack([right] * 3, axis=-1)
    out = SGMCpuEngine(SgmParams(max_disp=32)).predict(*rgb)
    assert out.values.shape == (24, 64)
    assert out.values.dtype == np.float32
    assert out.runtime_ms > 0


def test_cuda_engine_unavailable_without_extension(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cuda_sgm, "_load", lambda: None)
    engine = CudaSGMEngine()
    assert engine.requires_gpu
    assert not engine.is_available()
    left, right = shifted_noise_pair(8, 8, shift=1)
    with pytest.raises(RuntimeError, match="CUDA"):
        engine.predict(left, right)
