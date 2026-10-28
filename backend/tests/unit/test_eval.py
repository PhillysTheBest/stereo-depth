import json
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pytest

from stereo_depth.engines.base import Image
from stereo_depth.eval.bench import main
from stereo_depth.eval.datasets import kitti2015, middlebury, read_pfm, write_pfm
from stereo_depth.eval.metrics import fill_holes, mean_scores, score
from tests.unit.conftest import SHIFT


def test_perfect_prediction_scores_zero() -> None:
    gt = np.full((4, 4), 10.0, dtype=np.float32)
    s = score(gt.copy(), gt, np.ones_like(gt, dtype=bool), runtime_ms=1.0)
    assert s.epe == 0 and s.d1_all == 0 and s.bad_2 == 0 and s.density == 1


def test_d1_needs_both_thresholds() -> None:
    gt = np.array([[100.0, 100.0, 10.0, 10.0]], dtype=np.float32)
    # errors 4 (4%, not D1), 6 (6%, D1), 2.5 (bad-2 only), 1 (fine)
    pred = np.array([[104.0, 106.0, 12.5, 11.0]], dtype=np.float32)
    s = score(pred, gt, np.ones_like(gt, dtype=bool), runtime_ms=0.0)
    assert s.epe == pytest.approx((4 + 6 + 2.5 + 1) / 4)
    assert s.d1_all == pytest.approx(0.25)
    assert s.bad_2 == pytest.approx(0.75)


def test_missing_predictions_are_background_filled() -> None:
    gt = np.full((1, 3), 20.0, dtype=np.float32)
    pred = np.array([[np.nan, 20.0, np.nan]], dtype=np.float32)
    s = score(pred, gt, np.ones_like(gt, dtype=bool), runtime_ms=0.0)
    assert s.density == pytest.approx(1 / 3)
    assert s.epe == 0.0


def test_fill_holes_takes_smaller_neighbour() -> None:
    pred = np.array([[np.nan, 5.0, np.nan, np.nan, 9.0, np.nan], [np.nan] * 6], dtype=np.float32)
    out = fill_holes(pred)
    assert out[0].tolist() == [5.0, 5.0, 5.0, 5.0, 9.0, 9.0]
    assert out[1].tolist() == [0.0] * 6
    assert np.isnan(pred[0, 0])  # input untouched


def test_mask_and_zero_gt_are_ignored() -> None:
    gt = np.array([[10.0, 0.0, 10.0]], dtype=np.float32)
    pred = np.array([[10.0, 50.0, 99.0]], dtype=np.float32)
    s = score(pred, gt, np.array([[True, True, False]]), runtime_ms=0.0)
    assert s.epe == 0.0
    with pytest.raises(ValueError):
        score(pred, gt, np.zeros_like(gt, dtype=bool), runtime_ms=0.0)


def test_mean_scores() -> None:
    gt = np.full((2, 2), 10.0, dtype=np.float32)
    a = score(gt, gt, np.ones_like(gt, dtype=bool), 2.0)
    b = score(gt + 4, gt, np.ones_like(gt, dtype=bool), 4.0)
    m = mean_scores([a, b])
    assert m["epe"] == pytest.approx(2.0)
    assert m["runtime_ms"] == pytest.approx(3.0)


def test_pfm_round_trip(tmp_path: Path) -> None:
    arr = np.arange(6, dtype=np.float32).reshape(2, 3)
    arr[0, 0] = np.inf
    write_pfm(tmp_path / "a.pfm", arr)
    out = read_pfm(tmp_path / "a.pfm")
    assert np.isnan(out[0, 0])
    assert np.array_equal(out[1], arr[1])


def _fake_kitti(root: Path, pair: tuple[Image, Image]) -> None:
    base = root / "training"
    for d in ("image_2", "image_3", "disp_occ_0"):
        (base / d).mkdir(parents=True)
    left, right = pair
    gt = np.full(left.shape[:2], SHIFT * 256, dtype=np.uint16)
    gt[:, :140] = 0  # SGBM leaves the first numDisparities columns empty
    for i in range(2):
        iio.imwrite(base / "image_2" / f"00000{i}_10.png", left)
        iio.imwrite(base / "image_3" / f"00000{i}_10.png", right)
        iio.imwrite(base / "disp_occ_0" / f"00000{i}_10.png", gt)


def _fake_middlebury(root: Path, pair: tuple[Image, Image]) -> None:
    scene = root / "trainingQ" / "Toy"
    scene.mkdir(parents=True)
    left, right = pair
    iio.imwrite(scene / "im0.png", left)
    iio.imwrite(scene / "im1.png", right)
    gt = np.full(left.shape[:2], SHIFT, dtype=np.float32)
    gt[:, :140] = np.inf
    write_pfm(scene / "disp0GT.pfm", gt)
    iio.imwrite(scene / "mask0nocc.png", np.full(left.shape[:2], 255, dtype=np.uint8))
    (scene / "calib.txt").write_text("cam0=[1 0 0; 0 1 0; 0 0 1]\nndisp=32\n")


def test_kitti_loader(tmp_path: Path, shifted_pair: tuple[Image, Image]) -> None:
    _fake_kitti(tmp_path, shifted_pair)
    samples = list(kitti2015(tmp_path))
    assert [s.name for s in samples] == ["000000_10", "000001_10"]
    assert samples[0].gt[0, 160] == SHIFT
    assert not samples[0].valid[0, 0]


def test_middlebury_loader(tmp_path: Path, shifted_pair: tuple[Image, Image]) -> None:
    _fake_middlebury(tmp_path, shifted_pair)
    (sample,) = middlebury(tmp_path)
    assert sample.name == "Toy"
    assert sample.max_disp == 32
    assert not sample.valid[0, 0] and sample.valid[0, 160]


def test_loaders_explain_missing_data(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="download_kitti"):
        next(kitti2015(tmp_path))
    with pytest.raises(FileNotFoundError, match="download_middlebury"):
        next(middlebury(tmp_path))


@pytest.mark.parametrize("dataset", ["kitti2015", "middlebury"])
def test_bench_cli_end_to_end(
    dataset: str,
    tmp_path: Path,
    shifted_pair: tuple[Image, Image],
    capsys: pytest.CaptureFixture[str],
) -> None:
    (_fake_kitti if dataset == "kitti2015" else _fake_middlebury)(tmp_path, shifted_pair)
    out = tmp_path / "result.json"
    assert main(["--dataset", dataset, "--root", str(tmp_path), "--out", str(out)]) == 0
    result = json.loads(out.read_text())
    assert result["config"]["engine"] == "sgbm"
    assert result["summary"]["d1_all"] < 0.1
    assert "mean" in capsys.readouterr().out


def test_bench_compares_several_engines(
    tmp_path: Path,
    shifted_pair: tuple[Image, Image],
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from stereo_depth.engines import SGBMEngine
    from stereo_depth.eval import bench

    engines = {"sgbm": SGBMEngine(), "sgbm_wide": SGBMEngine(num_disparities=64, block_size=7)}
    monkeypatch.setattr(bench, "get_engines", lambda: engines)
    _fake_middlebury(tmp_path, shifted_pair)
    args = ["--dataset", "middlebury", "--root", str(tmp_path), "--engine", "sgbm", "sgbm_wide"]
    assert main(args) == 0
    table = capsys.readouterr().out
    assert "| engine | EPE | D1-all | bad-2.0 | ms/pair |" in table
    assert "| sgbm |" in table and "| sgbm_wide |" in table
