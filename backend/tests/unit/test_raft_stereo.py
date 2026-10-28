from pathlib import Path

import numpy as np
import pytest

from stereo_depth.config import Settings
from stereo_depth.engines.raft_stereo import RAFTStereoEngine

torch = pytest.importorskip("torch")


def _random_weights() -> dict[str, object]:
    from stereo_depth.engines.raft_stereo._vendor.raft_stereo import RAFTStereo
    from stereo_depth.engines.raft_stereo.engine import MODEL_ARGS

    torch.manual_seed(0)
    # Prefix keys the way the released nn.DataParallel checkpoints do.
    return {f"module.{k}": v for k, v in RAFTStereo(MODEL_ARGS).state_dict().items()}


def _engine(**overrides: int) -> RAFTStereoEngine:
    settings = Settings(raft_cpu_iters=2, **overrides)  # type: ignore[arg-type]
    engine = RAFTStereoEngine(settings, checkpoint=Path("/nonexistent.pth"), device="cpu")
    engine.load(_random_weights())
    return engine


def test_unavailable_without_weights() -> None:
    engine = RAFTStereoEngine(Settings(), checkpoint=Path("/nonexistent.pth"))
    assert not engine.is_available()
    assert not engine.loaded


@pytest.mark.parametrize("max_px", [640, 40])
def test_disparity_matches_input_size(max_px: int) -> None:
    engine = _engine(raft_cpu_max_px=max_px)
    assert engine.loaded and engine.is_available()
    rng = np.random.default_rng(0)
    left = rng.integers(0, 255, (50, 70, 3), dtype=np.uint8)
    right = np.roll(left, -4, axis=1)
    out = engine.predict(left, right)
    assert out.values.shape == (50, 70)
    assert out.values.dtype == np.float32
    assert out.runtime_ms > 0
    assert not (out.values <= 0).any()  # non-positive disparity is reported as NaN


def test_rejects_mismatched_pair() -> None:
    engine = _engine()
    with pytest.raises(ValueError, match="differ in size"):
        engine.predict(np.zeros((8, 8, 3), np.uint8), np.zeros((8, 9, 3), np.uint8))
