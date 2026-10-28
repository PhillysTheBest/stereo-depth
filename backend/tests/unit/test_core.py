import io

import numpy as np
import pytest
from plyfile import PlyData

from stereo_depth.core.color import colorize
from stereo_depth.core.depth import disparity_to_depth
from stereo_depth.core.io import decode_image, encode_png, load_image
from stereo_depth.core.ply import to_ply_bytes
from stereo_depth.core.resize import downscale, upscale_disparity


def test_png_round_trip(tmp_path) -> None:  # type: ignore[no-untyped-def]
    img = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
    data = encode_png(img)
    assert np.array_equal(decode_image(data), img)
    path = tmp_path / "x.png"
    path.write_bytes(data)
    assert np.array_equal(load_image(path), img)


def test_decode_grayscale_and_rgba() -> None:
    gray = np.full((3, 4), 7, dtype=np.uint8)
    assert decode_image(encode_png(gray)).shape == (3, 4, 3)
    rgba = np.zeros((3, 4, 4), dtype=np.uint8)
    assert decode_image(encode_png(rgba)).shape == (3, 4, 3)


def test_decode_rejects_garbage() -> None:
    with pytest.raises(ValueError):
        decode_image(b"not an image")


def test_downscale_limits_long_side() -> None:
    img = np.zeros((1000, 2560, 3), dtype=np.uint8)
    out, scale = downscale(img, 1280)
    assert out.shape == (500, 1280, 3)
    assert scale == pytest.approx(0.5)


def test_downscale_leaves_small_images() -> None:
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    out, scale = downscale(img, 1280)
    assert out is img
    assert scale == 1.0


def test_upscale_disparity_converts_units() -> None:
    disp = np.full((50, 64), 10.0, dtype=np.float32)
    out = upscale_disparity(disp, (100, 128), 0.5)
    assert out.shape == (100, 128)
    assert np.allclose(out, 20.0)


def test_disparity_to_depth() -> None:
    disp = np.array([[10.0, 0.0, np.nan, -1.0]], dtype=np.float32)
    depth = disparity_to_depth(disp, focal_px=700.0, baseline_m=0.1)
    assert depth[0, 0] == pytest.approx(7.0)
    assert np.isnan(depth[0, 1:]).all()
    with pytest.raises(ValueError):
        disparity_to_depth(disp, focal_px=0.0, baseline_m=0.1)


def test_ply_export_skips_invalid() -> None:
    depth = np.array([[1.0, np.nan], [2.0, 4.0]], dtype=np.float32)
    colors = np.full((2, 2, 3), 200, dtype=np.uint8)
    ply = PlyData.read(io.BytesIO(to_ply_bytes(depth, colors, focal_px=1.0)))
    vertex = ply["vertex"]
    assert vertex.count == 3
    assert sorted(vertex["z"].tolist()) == [1.0, 2.0, 4.0]
    assert set(vertex["red"].tolist()) == {200}


def test_ply_rejects_mismatched_colors() -> None:
    with pytest.raises(ValueError):
        to_ply_bytes(np.ones((2, 2), np.float32), np.zeros((3, 2, 3), np.uint8), 1.0)


def test_colorize_blacks_out_nan() -> None:
    values = np.array([[0.0, 1.0, np.nan]], dtype=np.float32)
    rgb = colorize(values)
    assert rgb.shape == (1, 3, 3)
    assert rgb.dtype == np.uint8
    assert (rgb[0, 2] == 0).all()
    assert not np.array_equal(rgb[0, 0], rgb[0, 1])
    assert colorize(np.full((2, 2), np.nan, np.float32)).sum() == 0
