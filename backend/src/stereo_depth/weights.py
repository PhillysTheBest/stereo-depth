"""Fetch pretrained weights into ``<data_dir>/weights``.

Run ``python -m stereo_depth.weights`` (or ``scripts/fetch_weights.py`` from the repo root).
Weights come from the Hugging Face repo in ``STEREO_RAFT_HF_REPO`` when it is set, otherwise from
the official RAFT-Stereo release archive.
"""

import argparse
import hashlib
import io
import logging
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

from stereo_depth.config import Settings

log = logging.getLogger(__name__)

# The authors' release (README of princeton-vl/RAFT-Stereo); holds every checkpoint under models/.
OFFICIAL_ZIP_URL = (
    "https://www.dropbox.com/scl/fi/5khx1bhz84dapi8vtwapg/models.zip"
    "?rlkey=ggddrn1du1iiq6mgc2dsdpmwi&dl=1"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_from_hf(repo_id: str, filename: str, dest: Path) -> Path:
    from huggingface_hub import hf_hub_download

    cached = Path(hf_hub_download(repo_id=repo_id, filename=filename))
    shutil.copyfile(cached, dest)
    return dest


def fetch_from_official_zip(filename: str, dest: Path, url: str = OFFICIAL_ZIP_URL) -> Path:
    with urllib.request.urlopen(url, timeout=120) as response:  # noqa: S310 (fixed https URL)
        archive = zipfile.ZipFile(io.BytesIO(response.read()))
    member = next((n for n in archive.namelist() if Path(n).name == filename), None)
    if member is None:
        raise FileNotFoundError(f"{filename} is not in the release archive")
    dest.write_bytes(archive.read(member))
    return dest


def fetch_raft_stereo(settings: Settings, source: str = "auto", force: bool = False) -> Path:
    dest = Path(settings.weights_dir) / settings.raft_checkpoint
    if dest.is_file() and not force:
        log.info("%s already present", dest)
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    if source == "auto":
        source = "hf" if settings.raft_hf_repo else "official"
    if source == "hf":
        if not settings.raft_hf_repo:
            raise ValueError("set STEREO_RAFT_HF_REPO to fetch from Hugging Face")
        fetch_from_hf(settings.raft_hf_repo, settings.raft_checkpoint, dest)
    else:
        fetch_from_official_zip(settings.raft_checkpoint, dest)
    log.info("wrote %s (sha256 %s)", dest, sha256(dest))
    return dest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", choices=["auto", "hf", "official"], default="auto")
    parser.add_argument("--force", action="store_true", help="download even if the file exists")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        fetch_raft_stereo(Settings(), source=args.source, force=args.force)
    except Exception as exc:  # report and exit non-zero; the API still serves the other engines
        log.error("could not fetch RAFT-Stereo weights: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
