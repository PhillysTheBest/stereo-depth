"""The half of the bridge that runs inside a Colab notebook.

`launch()` starts an HTTP server on 127.0.0.1 and exposes it through a Cloudflare
quick tunnel, using a pinned cloudflared release whose checksum is verified before
it runs. Commands run as background jobs that the client polls, because a quick
tunnel drops any single request that takes longer than about 100 seconds.
"""

from __future__ import annotations

import base64
import hashlib
import os
import re
import secrets
import shutil
import subprocess
import tempfile
import threading
import time
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import IO, Any

from colab_bridge import __version__
from colab_bridge.protocol import (
    MAX_AGE_S,
    MAX_BODY_BYTES,
    MAX_FILE_BYTES,
    BridgeError,
    make_uri,
    new_key,
    open_sealed,
    seal,
)

CLOUDFLARED_VERSION = "2026.9.3"
CLOUDFLARED_SHA256 = "77e26d8d900e0b8469f416239d14b5f296525fdf79fee6f511ef55609e3fbac2"
CLOUDFLARED_URL = (
    "https://github.com/cloudflare/cloudflared/releases/download/"
    f"{CLOUDFLARED_VERSION}/cloudflared-linux-amd64"
)
POLL_CHUNK_BYTES = 256 * 1024


@dataclass
class Job:
    proc: subprocess.Popen[bytes]
    log_path: Path


@dataclass
class Bridge:
    """Request handling, kept apart from HTTP so it can be tested directly."""

    key: str
    workdir: Path
    started: float = field(default_factory=time.time)
    jobs: dict[str, Job] = field(default_factory=dict)
    seen: dict[bytes, float] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def handle(self, route: str, body: bytes) -> bytes:
        request = open_sealed(self.key, body)
        self._reject_replay(body)
        handler = self._routes().get(route)
        if handler is None:
            raise ValueError(f"unknown route {route}")
        return seal(self.key, handler(request))

    def _reject_replay(self, body: bytes) -> None:
        digest = hashlib.sha256(body).digest()
        now = time.time()
        with self.lock:
            self.seen = {d: t for d, t in self.seen.items() if t > now}
            if digest in self.seen:
                raise BridgeError("replayed message")
            # A token stays acceptable for MAX_AGE_S after creation; allow for clock skew.
            self.seen[digest] = now + 2 * MAX_AGE_S

    def _routes(self) -> dict[str, Callable[[dict[str, Any]], dict[str, Any]]]:
        return {
            "/health": self._health,
            "/start": self._start,
            "/poll": self._poll,
            "/kill": self._kill,
            "/put": self._put,
            "/get": self._get,
        }

    def _health(self, _: dict[str, Any]) -> dict[str, Any]:
        try:
            query = "--query-gpu=name,memory.used,memory.total"
            gpu = subprocess.run(
                ["nvidia-smi", query, "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=30,
            ).stdout.strip()
        except (FileNotFoundError, subprocess.TimeoutExpired):
            gpu = ""
        return {
            "version": __version__,
            "gpu": gpu or "none",
            "disk_free_gb": round(shutil.disk_usage(self.workdir).free / 1e9, 1),
            "uptime_s": int(time.time() - self.started),
        }

    def _start(self, req: dict[str, Any]) -> dict[str, Any]:
        job_id = secrets.token_hex(8)
        log_path = self.workdir / f"job-{job_id}.log"
        cwd = req.get("cwd") or str(self.workdir)
        with log_path.open("wb") as log:
            proc = subprocess.Popen(
                req["command"],
                shell=True,
                cwd=cwd,
                stdout=log,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
        with self.lock:
            self.jobs[job_id] = Job(proc, log_path)
        return {"job": job_id}

    def _job(self, req: dict[str, Any]) -> Job:
        with self.lock:
            job = self.jobs.get(req["job"])
        if job is None:
            raise ValueError("unknown job")
        return job

    def _poll(self, req: dict[str, Any]) -> dict[str, Any]:
        job = self._job(req)
        # Read the exit code first so no output written before exit is missed.
        exit_code = job.proc.poll()
        offset = int(req.get("offset", 0))
        with job.log_path.open("rb") as log:
            log.seek(offset)
            chunk = log.read(POLL_CHUNK_BYTES)
        more = len(chunk) == POLL_CHUNK_BYTES
        return {
            "output": chunk.decode(errors="replace"),
            "offset": offset + len(chunk),
            "done": exit_code is not None and not more,
            "exit_code": exit_code,
        }

    def _kill(self, req: dict[str, Any]) -> dict[str, Any]:
        job = self._job(req)
        if job.proc.poll() is None:
            os.killpg(job.proc.pid, 9)
        return {"killed": True}

    def _put(self, req: dict[str, Any]) -> dict[str, Any]:
        data = base64.b64decode(req["data"])
        if len(data) > MAX_FILE_BYTES:
            raise ValueError("file too large")
        path = Path(req["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return {"path": str(path), "bytes": len(data)}

    def _get(self, req: dict[str, Any]) -> dict[str, Any]:
        path = Path(req["path"])
        if path.stat().st_size > MAX_FILE_BYTES:
            raise ValueError("file too large")
        return {"data": base64.b64encode(path.read_bytes()).decode()}


def make_http_server(bridge: Bridge, port: int = 0) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 (http.server naming)
            length = int(self.headers.get("Content-Length", 0))
            if length > MAX_BODY_BYTES:
                self._reply(413, b"")
                return
            body = self.rfile.read(length)
            try:
                reply = bridge.handle(self.path, body)
            except BridgeError:
                # Say nothing more to a caller that may not hold the key.
                self._reply(403, b"")
                return
            except Exception as e:
                # Past authentication, so the caller holds the key: tell them what broke.
                self._reply(500, seal(bridge.key, {"error": f"{type(e).__name__}: {e}"}))
                return
            self._reply(200, reply)

        def _reply(self, status: int, body: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:
            pass

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def _drain(stream: IO[str]) -> None:
    for _ in stream:
        pass


def fetch_cloudflared(dest: Path) -> Path:
    if not dest.exists():
        tmp = dest.with_suffix(".part")
        urllib.request.urlretrieve(CLOUDFLARED_URL, tmp)
        tmp.rename(dest)
    digest = hashlib.sha256(dest.read_bytes()).hexdigest()
    if digest != CLOUDFLARED_SHA256:
        dest.unlink()
        raise RuntimeError(f"cloudflared checksum mismatch: got {digest}")
    dest.chmod(0o755)
    return dest


def launch(workdir: str = "/content/colab_bridge", tunnel_timeout_s: int = 60) -> str:
    """Start the server and tunnel in the background and return the connection string."""
    work = Path(workdir)
    work.mkdir(parents=True, exist_ok=True)
    bridge = Bridge(key=new_key(), workdir=work)
    httpd = make_http_server(bridge)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    port = httpd.server_address[1]

    binary = fetch_cloudflared(Path(tempfile.gettempdir()) / f"cloudflared-{CLOUDFLARED_VERSION}")
    tunnel = subprocess.Popen(
        [str(binary), "tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{port}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert tunnel.stderr is not None
    deadline = time.time() + tunnel_timeout_s
    for line in tunnel.stderr:
        match = re.search(r"https://([a-z0-9-]+\.trycloudflare\.com)", line)
        if match:
            # Keep draining cloudflared's log so its pipe never fills up.
            threading.Thread(target=_drain, args=(tunnel.stderr,), daemon=True).start()
            return make_uri(bridge.key, match.group(1))
        if time.time() > deadline:
            break
    tunnel.kill()
    httpd.shutdown()
    raise RuntimeError("cloudflared did not report a tunnel URL")
