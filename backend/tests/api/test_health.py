from fastapi.testclient import TestClient

from stereo_depth.api.app import app
from stereo_depth.engines import DepthEngine


def test_health_ok() -> None:
    response = TestClient(app).get("/v1/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert isinstance(body["gpu"], bool)
    assert any(e["name"] == "raft_stereo" for e in body["engines"])


def test_health_reports_model_loaded(fake_engines: dict[str, DepthEngine]) -> None:
    body = TestClient(app).get("/v1/health").json()
    by_name = {e["name"]: e for e in body["engines"]}
    assert by_name["constant"]["loaded"] is True
    assert by_name["missing"]["available"] is False
