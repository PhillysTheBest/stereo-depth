"""Write the API's OpenAPI schema. Usage: python -m stereo_depth.api.openapi [path]

The committed copy at ``backend/openapi.json`` is what the frontend generates its client from;
a test fails when it drifts from the app.
"""

import json
import sys
from pathlib import Path

from stereo_depth.api.app import app

DEFAULT_PATH = Path(__file__).resolve().parents[3] / "openapi.json"


def render() -> str:
    return json.dumps(app.openapi(), indent=2, sort_keys=True) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    path = Path(args[0]) if args else DEFAULT_PATH
    path.write_text(render())
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
