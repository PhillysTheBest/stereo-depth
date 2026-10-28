"""colab-bridge command line.

The connection string is read from stdin or the COLAB_BRIDGE_URI environment
variable, never from argv, so it stays out of shell history and process lists.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from colab_bridge.client import Client
from colab_bridge.protocol import BridgeError, parse_uri

CONFIG_PATH = Path.home() / ".config" / "colab-bridge" / "connection"
REMOTE_RUNS = "/content/colab_bridge/runs"


def load_client() -> Client:
    uri = os.environ.get("COLAB_BRIDGE_URI")
    if not uri and CONFIG_PATH.exists():
        uri = CONFIG_PATH.read_text()
    if not uri:
        raise BridgeError("not connected: run `colab-bridge connect` and paste the string")
    return Client(*parse_uri(uri))


def save_uri(uri: str) -> None:
    parse_uri(uri)
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(CONFIG_PATH, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(uri.strip())


def set_params(notebook: dict[str, Any], params: dict[str, str]) -> dict[str, Any]:
    """Replace the first top-level `NAME = ...` line for each param, papermill style."""
    pending = dict(params)
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = cell["source"]
        lines = (
            "".join(source).splitlines(keepends=True)
            if isinstance(source, list)
            else (source.splitlines(keepends=True))
        )
        for i, line in enumerate(lines):
            match = re.match(r"([A-Za-z_]\w*)\s*=", line)
            if match and match.group(1) in pending:
                end = "\n" if line.endswith("\n") else ""
                lines[i] = f"{match.group(1)} = {pending.pop(match.group(1))!r}{end}"
        cell["source"] = lines
    if pending:
        raise BridgeError(f"no `NAME = ...` line found for: {', '.join(pending)}")
    return notebook


def notebook_errors(notebook: dict[str, Any]) -> list[str]:
    return [
        f"{out.get('ename')}: {out.get('evalue')}"
        for cell in notebook["cells"]
        for out in cell.get("outputs", [])
        if out.get("output_type") == "error"
    ]


def run_notebook(client: Client, path: Path, params: dict[str, str], out: Path) -> int:
    notebook = set_params(json.loads(path.read_text()), params)
    remote_dir = f"{REMOTE_RUNS}/{path.stem}"
    remote_in = f"{remote_dir}/{path.name}"
    local_tmp = out.with_suffix(".input.ipynb")
    local_tmp.write_text(json.dumps(notebook, indent=1))
    try:
        client.put(local_tmp, remote_in)
    finally:
        local_tmp.unlink()
    command = (
        "jupyter nbconvert --to notebook --execute --allow-errors "
        f"--ExecutePreprocessor.timeout=-1 --output executed.ipynb {path.name}"
    )
    code = client.run(command, cwd=remote_dir, on_output=_echo)
    if code != 0:
        return code
    client.get(f"{remote_dir}/executed.ipynb", out)
    print(f"executed notebook saved to {out}")
    errors = notebook_errors(json.loads(out.read_text()))
    for error in errors:
        print(f"cell error: {error}", file=sys.stderr)
    return 1 if errors else 0


def _echo(text: str) -> None:
    sys.stdout.write(text)
    sys.stdout.flush()


def _param(text: str) -> tuple[str, str]:
    name, sep, value = text.partition("=")
    if not sep:
        raise argparse.ArgumentTypeError("use NAME=VALUE")
    return name, value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="colab-bridge", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("connect", help="save the connection string read from stdin")
    sub.add_parser("disconnect", help="forget the saved connection string")
    sub.add_parser("health", help="show GPU, disk and uptime")
    p = sub.add_parser("run", help="run a shell command on Colab and stream its output")
    p.add_argument("command")
    p.add_argument("--cwd")
    p.add_argument("--timeout", type=float, help="kill the command after this many seconds")
    p = sub.add_parser("put", help="upload a file (50 MB max)")
    p.add_argument("local")
    p.add_argument("remote")
    p = sub.add_parser("get", help="download a file (50 MB max)")
    p.add_argument("remote")
    p.add_argument("local")
    p = sub.add_parser("notebook", help="execute a notebook on Colab and download the result")
    p.add_argument("path", type=Path)
    p.add_argument("--set", type=_param, action="append", default=[], metavar="NAME=VALUE")
    p.add_argument("--out", type=Path, help="default: <name>.executed.ipynb next to the input")
    args = parser.parse_args(argv)

    try:
        if args.cmd == "connect":
            save_uri(sys.stdin.readline())
            print(f"saved to {CONFIG_PATH}")
            return 0
        if args.cmd == "disconnect":
            CONFIG_PATH.unlink(missing_ok=True)
            return 0
        client = load_client()
        if args.cmd == "health":
            print(json.dumps(client.health(), indent=2))
            return 0
        if args.cmd == "run":
            return client.run(args.command, args.cwd, _echo, timeout_s=args.timeout)
        if args.cmd == "put":
            print(f"uploaded {client.put(args.local, args.remote)} bytes")
            return 0
        if args.cmd == "get":
            print(f"downloaded {client.get(args.remote, args.local)} bytes")
            return 0
        out = args.out or args.path.with_name(f"{args.path.stem}.executed.ipynb")
        return run_notebook(client, args.path, dict(args.set), out)
    except BridgeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
