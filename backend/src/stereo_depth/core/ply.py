import io

import numpy as np
import numpy.typing as npt
from plyfile import PlyData, PlyElement

from stereo_depth.engines.base import Image


def depth_to_points(
    depth: npt.NDArray[np.float32],
    focal_px: float,
    cx: float | None = None,
    cy: float | None = None,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.bool_]]:
    """Back-project a depth map to camera-frame XYZ. Returns (N x 3 points, HxW validity mask)."""
    h, w = depth.shape
    cx = (w - 1) / 2 if cx is None else cx
    cy = (h - 1) / 2 if cy is None else cy
    valid = np.isfinite(depth) & (depth > 0)
    v, u = np.nonzero(valid)
    z = depth[valid]
    x = (u - cx) * z / focal_px
    y = (v - cy) * z / focal_px
    return np.stack([x, y, z], axis=-1).astype(np.float32), valid


def to_ply_bytes(
    depth: npt.NDArray[np.float32],
    colors: Image,
    focal_px: float,
    cx: float | None = None,
    cy: float | None = None,
) -> bytes:
    """Binary little-endian PLY of the coloured point cloud. NaN depth pixels are skipped."""
    if colors.shape[:2] != depth.shape:
        raise ValueError("colors and depth must have the same height and width")
    points, valid = depth_to_points(depth, focal_px, cx, cy)
    rgb = colors[valid]
    fields = [("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")]
    vertex = np.empty(len(points), dtype=fields)
    vertex["x"], vertex["y"], vertex["z"] = points[:, 0], points[:, 1], points[:, 2]
    vertex["red"], vertex["green"], vertex["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    buf = io.BytesIO()
    PlyData([PlyElement.describe(vertex, "vertex")], byte_order="<").write(buf)
    return buf.getvalue()
