import pytest
from fastapi.testclient import TestClient

from stereo_depth.api import app as app_module
from stereo_depth.api.app import app
from stereo_depth.engines import DepthEngine


def test_gpu_engines_hidden_on_cpu_host(fake_engines: dict[str, DepthEngine]) -> None:
    names = [e["name"] for e in TestClient(app).get("/v1/engines").json()]
    assert names == ["constant", "missing"]


def test_gpu_engines_listed_on_gpu_host(
    fake_engines: dict[str, DepthEngine], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(app_module, "_gpu_available", lambda: True)
    names = [e["name"] for e in TestClient(app).get("/v1/engines").json()]
    assert "gpu_only" in names


def test_real_registry_has_raft_stereo() -> None:
    engines = {e["name"]: e for e in TestClient(app).get("/v1/engines").json()}
    assert engines["raft_stereo"]["requires_gpu"] is False
