#!/usr/bin/env bash
# Middlebury v3 (MiddEval3) training images and ground truth.
# Resolution Q (quarter, default), H (half) or F (full). Q is about 70 MB.
set -euo pipefail

DEST="${1:-data/middlebury}"
RES="${2:-Q}"
BASE="https://vision.middlebury.edu/stereo/submit3/zip"

case "$RES" in Q|H|F) ;; *) echo "resolution must be Q, H or F" >&2; exit 1 ;; esac

mkdir -p "$DEST"
if [ -d "$DEST/training$RES" ] && ls "$DEST/training$RES"/*/disp0GT.pfm >/dev/null 2>&1; then
  echo "Middlebury $RES already in $DEST"
  exit 0
fi
for zip in "MiddEval3-data-$RES.zip" "MiddEval3-GT0-$RES.zip"; do
  curl -fL --retry 3 -o "$DEST/$zip" "$BASE/$zip"
  unzip -qo "$DEST/$zip" -d "$DEST"
  rm "$DEST/$zip"
done
# The zips unpack into MiddEval3/; flatten so the loader sees $DEST/training$RES.
if [ -d "$DEST/MiddEval3" ]; then
  cp -r "$DEST/MiddEval3/." "$DEST/"
  rm -rf "$DEST/MiddEval3"
fi
echo "Middlebury $RES ready in $DEST"
