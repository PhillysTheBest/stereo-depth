#!/usr/bin/env bash
# KITTI 2015 stereo (scene flow) training + testing images, about 1.7 GB.
# Licence: CC BY-NC-SA 3.0, research use only. Never commit the data.
set -euo pipefail

DEST="${1:-data/kitti2015}"
URL="https://s3.eu-central-1.amazonaws.com/avg-kitti/data_scene_flow.zip"

mkdir -p "$DEST"
if [ -d "$DEST/training/image_2" ]; then
  echo "KITTI 2015 already in $DEST"
  exit 0
fi
curl -fL --retry 3 -o "$DEST/data_scene_flow.zip" "$URL"
unzip -q "$DEST/data_scene_flow.zip" -d "$DEST"
rm "$DEST/data_scene_flow.zip"
echo "KITTI 2015 ready in $DEST"
