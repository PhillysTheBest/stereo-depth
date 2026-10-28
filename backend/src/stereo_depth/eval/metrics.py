from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

Map = npt.NDArray[np.float32]


@dataclass(frozen=True)
class PairScores:
    """Scores for one pair. Errors are measured over pixels with ground truth.

    Pixels the engine left invalid (NaN) are filled by background interpolation first, as the
    KITTI devkit does, so scores are comparable with the leaderboard. density reports how many
    ground-truth pixels got a real prediction.
    """

    epe: float
    d1_all: float
    bad_2: float
    density: float
    runtime_ms: float


def fill_holes(pred: Map) -> Map:
    """KITTI devkit background interpolation: each run of NaN in a row takes the smaller of its
    two neighbouring valid disparities, and runs touching the image edge take the nearest one.
    Rows with no valid pixel become 0."""
    out = pred.copy()
    cols = np.arange(pred.shape[1])
    for row in out:
        ok = np.isfinite(row)
        if ok.all():
            continue
        if not ok.any():
            row[:] = 0.0
            continue
        idx = cols[ok]
        right = np.searchsorted(idx, cols)  # first valid column at or after each pixel
        left_i = idx[np.clip(right - 1, 0, len(idx) - 1)]
        right_i = idx[np.clip(right, 0, len(idx) - 1)]
        row[~ok] = np.minimum(row[left_i], row[right_i])[~ok]
    return out


def _errors(pred: Map, gt: Map, valid: npt.NDArray[np.bool_]) -> tuple[Map, Map]:
    p = fill_holes(pred)[valid]
    g = gt[valid]
    return np.abs(p - g).astype(np.float32), g


def score(pred: Map, gt: Map, valid: npt.NDArray[np.bool_], runtime_ms: float) -> PairScores:
    if pred.shape != gt.shape or gt.shape != valid.shape:
        raise ValueError(f"shape mismatch: pred {pred.shape}, gt {gt.shape}, mask {valid.shape}")
    valid = valid & np.isfinite(gt) & (gt > 0)
    n = int(valid.sum())
    if n == 0:
        raise ValueError("no valid ground-truth pixels")
    err, g = _errors(pred, gt, valid)
    return PairScores(
        epe=float(err.mean()),
        # KITTI 2015 outlier: error > 3 px and > 5% of the true disparity.
        d1_all=float(((err > 3.0) & (err > 0.05 * g)).mean()),
        bad_2=float((err > 2.0).mean()),
        density=float(np.isfinite(pred[valid]).mean()),
        runtime_ms=runtime_ms,
    )


def mean_scores(scores: list[PairScores]) -> dict[str, float]:
    if not scores:
        raise ValueError("no scores to average")
    fields = ("epe", "d1_all", "bad_2", "density", "runtime_ms")
    return {f: float(np.mean([getattr(s, f) for s in scores])) for f in fields}
