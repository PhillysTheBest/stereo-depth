"""The half of the bridge that runs wherever the agent is."""

from __future__ import annotations

import base64
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

from colab_bridge.protocol import MAX_FILE_BYTES, BridgeError, open_sealed, seal


class Client:
    def __init__(self, base_url: str, key: str, request_timeout_s: float = 90) -> None:
        self.base_url = base_url.rstrip("/")
        self.key = key
        self.request_timeout_s = request_timeout_s

    def call(self, route: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        req = urllib.request.Request(
            self.base_url + route,
            data=seal(self.key, payload or {}),
            headers={"Content-Type": "application/octet-stream"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.request_timeout_s) as resp:
                body = resp.read()
        except urllib.error.HTTPError as e:
            if e.code == 403:
                raise BridgeError("rejected: wrong connection string, or clock skew") from e
            if e.code == 500:
                raise BridgeError(open_sealed(self.key, e.read(), None)["error"]) from e
            raise BridgeError(f"HTTP {e.code}: is the Colab session still running?") from e
        except urllib.error.URLError as e:
            raise BridgeError(f"cannot reach Colab ({e.reason}): restart the notebook") from e
        # Responses are not replay-checked, but must decrypt under our key.
        return open_sealed(self.key, body, None)

    def health(self) -> dict[str, Any]:
        return self.call("/health")

    def start(self, command: str, cwd: str | None = None) -> str:
        return str(self.call("/start", {"command": command, "cwd": cwd})["job"])

    def kill(self, job: str) -> None:
        self.call("/kill", {"job": job})

    def run(
        self,
        command: str,
        cwd: str | None = None,
        on_output: Callable[[str], None] | None = None,
        poll_s: float = 2.0,
        timeout_s: float | None = None,
    ) -> int:
        """Run a shell command to completion, streaming its output; return the exit code."""
        job = self.start(command, cwd)
        offset = 0
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        while True:
            reply = self.call("/poll", {"job": job, "offset": offset})
            offset = reply["offset"]
            if reply["output"] and on_output:
                on_output(reply["output"])
            if reply["done"]:
                return int(reply["exit_code"])
            if deadline is not None and time.monotonic() > deadline:
                self.kill(job)
                raise BridgeError(f"command timed out after {timeout_s}s and was killed")
            if not reply["output"]:
                time.sleep(poll_s)

    def put(self, local: str | Path, remote: str) -> int:
        data = Path(local).read_bytes()
        if len(data) > MAX_FILE_BYTES:
            raise BridgeError("file over 50 MB: fetch it from inside Colab instead")
        reply = self.call("/put", {"path": remote, "data": base64.b64encode(data).decode()})
        return int(reply["bytes"])

    def get(self, remote: str, local: str | Path) -> int:
        data = base64.b64decode(self.call("/get", {"path": remote})["data"])
        dest = Path(local)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        return len(data)
