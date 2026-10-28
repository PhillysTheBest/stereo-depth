import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import numpy.typing as npt

from stereo_depth.core.io import load_image
from stereo_depth.engines.base import Image


@dataclass(frozen=True)
class StereoSample:
    name: str
    left: Image
    right: Image
    gt: npt.NDArray[np.float32]
    valid: npt.NDArray[np.bool_]
    max_disp: int | None = None


def read_pfm(path: str | Path) -> npt.NDArray[np.float32]:
    """Read a PFM file (Middlebury ground truth). Rows come out top-to-bottom; inf becomes NaN."""
    with open(path, "rb") as f:
        header = f.readline().strip()
        if header not in (b"Pf", b"PF"):
            raise ValueError(f"{path} is not a PFM file")
        channels = 1 if header == b"Pf" else 3
        dims = f.readline().strip()
        while dims.startswith(b"#"):
            dims = f.readline().strip()
        w, h = (int(x) for x in dims.split())
        scale = float(f.readline().strip())
        dtype = "<f4" if scale < 0 else ">f4"
        data = np.frombuffer(f.read(), dtype=dtype, count=w * h * channels)
    shape = (h, w) if channels == 1 else (h, w, 3)
    arr = np.flipud(data.reshape(shape)).astype(np.float32)
    arr[~np.isfinite(arr)] = np.nan
    return np.ascontiguousarray(arr)


def write_pfm(path: str | Path, arr: npt.NDArray[np.float32]) -> None:
    h, w = arr.shape
    with open(path, "wb") as f:
        f.write(f"Pf\n{w} {h}\n-1.0\n".encode())
        f.write(np.flipud(arr).astype("<f4").tobytes())


def kitti2015(root: str | Path, occluded: bool = True) -> Iterator[StereoSample]:
    """KITTI 2015 scene flow training split (200 pairs) from data_scene_flow.zip.

    Ground truth is 16-bit PNG disparity * 256, with 0 meaning no measurement. disp_occ_0
    (all pixels) is the official D1-all target; disp_noc_0 covers non-occluded pixels only.
    """
    base = Path(root) / "training"
    gt_dir = base / ("disp_occ_0" if occluded else "disp_noc_0")
    lefts = sorted((base / "image_2").glob("*_10.png"))
    if not lefts:
        raise FileNotFoundError(f"no KITTI 2015 images under {base}; run scripts/download_kitti.sh")
    for left_path in lefts:
        raw = np.asarray(iio.imread(gt_dir / left_path.name))
        gt = raw.astype(np.float32) / 256.0
        yield StereoSample(
            name=left_path.stem,
            left=load_image(left_path),
            right=load_image(base / "image_3" / left_path.name),
            gt=gt,
            valid=raw > 0,
        )


def _ndisp(calib: Path) -> int | None:
    if not calib.exists():
        return None
    match = re.search(r"^ndisp=(\d+)", calib.read_text(), flags=re.MULTILINE)
    return int(match.group(1)) if match else None


def middlebury(root: str | Path, resolution: str = "Q") -> Iterator[StereoSample]:
    """Middlebury v3 (MiddEval3) training scenes at Q, H or F resolution.

    Scored on non-occluded pixels (mask0nocc.png == 255), the usual bad-2.0 setting.
    """
    base = Path(root) / f"training{resolution}"
    scenes = sorted(p for p in base.glob("*") if (p / "im0.png").exists())
    if not scenes:
        raise FileNotFoundError(
            f"no Middlebury scenes under {base}; run scripts/download_middlebury.sh"
        )
    for scene in scenes:
        gt = read_pfm(scene / "disp0GT.pfm")
        mask_path = scene / "mask0nocc.png"
        valid = np.isfinite(gt)
        if mask_path.exists():
            valid &= np.asarray(iio.imread(mask_path)) == 255
        yield StereoSample(
            name=scene.name,
            left=load_image(scene / "im0.png"),
            right=load_image(scene / "im1.png"),
            gt=gt,
            valid=valid,
            max_disp=_ndisp(scene / "calib.txt"),
        )


DATASETS: dict[str, Callable[[Path], Iterator[StereoSample]]] = {
    "kitti2015": kitti2015,
    "middlebury": middlebury,
}
