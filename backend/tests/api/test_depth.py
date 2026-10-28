import base64
import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image as PILImage

from stereo_depth.api.app import app
from stereo_depth.engines import DepthEngine
from tests.api.conftest import png_bytes

pytestmark = pytest.mark.usefixtures("fake_engines")


def _post(client: TestClient, left: bytes, right: bytes, **form: str) -> dict[str, object]:
    files = {"left": ("l.png", left, "image/png"), "right": ("r.png", right, "image/png")}
    response = client.post("/v1/depth", files=files, data={"engine": "constant", **form})
    assert response.status_code == 200, response.text
    body: dict[str, object] = response.json()
    return body


def _decode_png(data_url: object) -> np.ndarray:
    assert isinstance(data_url, str) and data_url.startswith("data:image/png;base64,")
    raw = base64.b64decode(data_url.split(",", 1)[1])
    return np.asarray(PILImage.open(io.BytesIO(raw)))


def test_relative_depth_without_calibration() -> None:
    body = _post(TestClient(app), png_bytes(40, 30), png_bytes(40, 30, seed=1))
    assert body["unit"] == "relative"
    assert (body["width"], body["height"]) == (40, 30)
    assert _decode_png(body["disparity_png"]).shape == (30, 40, 3)
    assert _decode_png(body["depth_png"]).shape == (30, 40, 3)
    stats = body["depth_stats"]
    assert isinstance(stats, dict)
    assert stats["median"] == pytest.approx(1.0)
    assert stats["valid_fraction"] == pytest.approx(39 / 40)
    assert body["ply"] is None


def test_metric_depth_with_calibration() -> None:
    # Constant disparity of 8 px with f = 400 px and B = 0.1 m puts every pixel at 5 m.
    body = _post(
        TestClient(app), png_bytes(40, 30), png_bytes(40, 30), focal_px="400", baseline_m="0.1"
    )
    assert body["unit"] == "m"
    stats = body["depth_stats"]
    assert isinstance(stats, dict)
    assert stats["min"] == pytest.approx(5.0)
    assert stats["max"] == pytest.approx(5.0)


def test_downscaled_upload_keeps_metric_depth(monkeypatch: pytest.MonkeyPatch) -> None:
    # At half size the engine sees half the disparity for the same scene, and the focal length
    # in downscaled pixels halves too, so depth must not change.
    monkeypatch.setenv("STEREO_MAX_WIDTH_PX", "20")
    body = _post(
        TestClient(app), png_bytes(40, 30), png_bytes(40, 30), focal_px="800", baseline_m="0.1"
    )
    assert body["scale"] == pytest.approx(0.5)
    assert (body["width"], body["height"]) == (20, 15)
    stats = body["depth_stats"]
    assert isinstance(stats, dict)
    assert stats["median"] == pytest.approx(800 * 0.5 * 0.1 / 8)


def test_ply_on_request() -> None:
    body = _post(TestClient(app), png_bytes(8, 6), png_bytes(8, 6), include_ply="true")
    ply = body["ply"]
    assert isinstance(ply, str)
    raw = base64.b64decode(ply.split(",", 1)[1])
    assert raw.startswith(b"ply\n") and b"element vertex 42\n" in raw


def test_rejects_half_calibration() -> None:
    files = {"left": ("l.png", png_bytes(8, 6)), "right": ("r.png", png_bytes(8, 6))}
    response = TestClient(app).post(
        "/v1/depth", files=files, data={"engine": "constant", "focal_px": "400"}
    )
    assert response.status_code == 422


def test_rejects_unknown_and_gpu_engines() -> None:
    files = {"left": ("l.png", png_bytes(8, 6)), "right": ("r.png", png_bytes(8, 6))}
    client = TestClient(app)
    for name in ("nope", "gpu_only"):
        response = client.post("/v1/depth", files=files, data={"engine": name})
        assert response.status_code == 404


def test_engine_without_weights_is_503() -> None:
    files = {"left": ("l.png", png_bytes(8, 6)), "right": ("r.png", png_bytes(8, 6))}
    response = TestClient(app).post("/v1/depth", files=files, data={"engine": "missing"})
    assert response.status_code == 503


def test_rejects_bad_image_and_size_mismatch(fake_engines: dict[str, DepthEngine]) -> None:
    client = TestClient(app)
    bad = {"left": ("l.png", b"not an image"), "right": ("r.png", png_bytes(8, 6))}
    assert client.post("/v1/depth", files=bad, data={"engine": "constant"}).status_code == 400
    mismatch = {"left": ("l.png", png_bytes(8, 6)), "right": ("r.png", png_bytes(9, 6))}
    assert client.post("/v1/depth", files=mismatch, data={"engine": "constant"}).status_code == 400
    assert fake_engines["constant"].calls == 0  # type: ignore[attr-defined]
