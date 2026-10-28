"""Download pretrained weights into data/weights. Usage: python scripts/fetch_weights.py [--force]"""

import os
import subprocess
import sys
from pathlib import Path

backend = Path(__file__).resolve().parent.parent / "backend"
env = {**os.environ, "STEREO_DATA_DIR": os.environ.get("STEREO_DATA_DIR", str(backend.parent / "data"))}
cmd = ["uv", "run", "--extra", "cpu", "python", "-m", "stereo_depth.weights", *sys.argv[1:]]
sys.exit(subprocess.call(cmd, cwd=backend, env=env))
