from __future__ import annotations

import json
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from cryptography.fernet import Fernet

from colab_bridge import cli
from colab_bridge.client import Client
from colab_bridge.protocol import BridgeError, make_uri, new_key, parse_uri, seal
from colab_bridge.server import Bridge, make_http_server


@pytest.fixture
def bridge(tmp_path: Path) -> Bridge:
    return Bridge(key=new_key(), workdir=tmp_path)


@pytest.fixture
def client(bridge: Bridge) -> Iterator[Client]:
    httpd = make_http_server(bridge)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield Client(f"http://127.0.0.1:{httpd.server_address[1]}", bridge.key)
    httpd.shutdown()


def test_uri_round_trip() -> None:
    key = new_key()
    assert parse_uri(make_uri(key, "abc-def.trycloudflare.com")) == (
        "https://abc-def.trycloudflare.com",
        key,
    )


@pytest.mark.parametrize(
    "uri", ["cc://x@host", "colab-bridge://host", "colab-bridge://notakey@host"]
)
def test_bad_uri(uri: str) -> None:
    with pytest.raises(BridgeError):
        parse_uri(uri)


def test_wrong_key_rejected(bridge: Bridge) -> None:
    with pytest.raises(BridgeError):
        bridge.handle("/health", seal(new_key(), {}))


def test_replay_rejected(bridge: Bridge) -> None:
    body = seal(bridge.key, {})
    bridge.handle("/health", body)
    with pytest.raises(BridgeError, match="replayed"):
        bridge.handle("/health", body)


def test_stale_rejected(bridge: Bridge) -> None:
    old = Fernet(bridge.key.encode()).encrypt_at_time(b"{}", int(time.time()) - 3600)
    with pytest.raises(BridgeError):
        bridge.handle("/health", old)


def test_wrong_key_gets_bare_403(client: Client) -> None:
    intruder = Client(client.base_url, new_key())
    with pytest.raises(BridgeError, match="rejected"):
        intruder.health()


def test_health(client: Client) -> None:
    assert client.health()["version"] == "0.1.0"


def test_run_streams_output_and_exit_code(client: Client) -> None:
    chunks: list[str] = []
    code = client.run("echo hello; echo oops >&2; exit 3", on_output=chunks.append, poll_s=0.05)
    assert code == 3
    assert "hello" in "".join(chunks) and "oops" in "".join(chunks)


def test_run_large_output_is_complete(client: Client) -> None:
    chunks: list[str] = []
    code = client.run(f"{sys.executable} -c \"print('x' * 600000)\"", on_output=chunks.append)
    assert code == 0
    assert "".join(chunks).count("x") == 600000


def test_run_timeout_kills(client: Client) -> None:
    with pytest.raises(BridgeError, match="timed out"):
        client.run("sleep 30", poll_s=0.05, timeout_s=0.3)


def test_put_get(client: Client, tmp_path: Path) -> None:
    src = tmp_path / "a.bin"
    src.write_bytes(bytes(range(256)))
    assert client.put(src, str(tmp_path / "remote" / "a.bin")) == 256
    assert client.get(str(tmp_path / "remote" / "a.bin"), tmp_path / "b.bin") == 256
    assert (tmp_path / "b.bin").read_bytes() == bytes(range(256))


def test_server_error_is_reported(client: Client, tmp_path: Path) -> None:
    with pytest.raises(BridgeError, match="FileNotFoundError"):
        client.get(str(tmp_path / "missing"), tmp_path / "x")


def test_set_params() -> None:
    nb = {
        "cells": [
            {"cell_type": "markdown", "source": ["REF = 'no'\n"]},
            {"cell_type": "code", "source": ['REF = "main"\n', "print(REF)"]},
        ]
    }
    out = cli.set_params(nb, {"REF": "feature/x"})
    assert out["cells"][1]["source"] == ["REF = 'feature/x'\n", "print(REF)"]
    assert out["cells"][0]["source"] == ["REF = 'no'\n"]
    with pytest.raises(BridgeError, match="MISSING"):
        cli.set_params(nb, {"MISSING": "1"})


def test_notebook_errors() -> None:
    nb = {"cells": [{"outputs": [{"output_type": "error", "ename": "E", "evalue": "v"}]}]}
    assert cli.notebook_errors(nb) == ["E: v"]


def test_saved_uri_is_private(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "CONFIG_PATH", tmp_path / "c" / "connection")
    cli.save_uri(make_uri(new_key(), "h.trycloudflare.com") + "\n")
    assert (tmp_path / "c" / "connection").stat().st_mode & 0o077 == 0
    assert json.dumps(cli.CONFIG_PATH.read_text()).endswith('com"')
