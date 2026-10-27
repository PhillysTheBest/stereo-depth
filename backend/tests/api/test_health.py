from fastapi.testclient import TestClient

from stereo_depth.api.app import app


def test_health_ok() -> None:
    response = TestClient(app).get("/v1/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert isinstance(body["gpu"], bool)
