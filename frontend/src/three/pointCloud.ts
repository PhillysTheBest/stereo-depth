/** Pinhole intrinsics in pixels. cx and cy default to the image centre, fy to fx. */
export interface Intrinsics {
  fx: number;
  fy?: number;
  cx?: number;
  cy?: number;
}

export interface PointCloud {
  /** xyz per point, in the depth map's units. Three.js axes: x right, y up, camera looks down -z. */
  positions: Float32Array;
  /** rgb per point in 0..1. */
  colors: Float32Array;
  count: number;
}

export interface DepthToPointsOptions {
  /** RGBA pixels the same size as the depth map, for example ImageData.data. Grey when missing. */
  rgba?: Uint8ClampedArray;
  /** Keep every `step`th pixel in each direction, to thin large clouds. */
  step?: number;
}

const GREY = 0.7;

/** Back-projects a row-major depth map into 3D. Pixels with NaN, infinite or non-positive depth are skipped. */
export function depthToPoints(
  depth: Float32Array,
  width: number,
  height: number,
  intrinsics: Intrinsics,
  { rgba, step = 1 }: DepthToPointsOptions = {},
): PointCloud {
  if (depth.length !== width * height) {
    throw new Error(`depth has ${depth.length} values, expected ${width}x${height}`);
  }
  if (rgba && rgba.length !== width * height * 4) {
    throw new Error("rgba must be the same size as the depth map");
  }
  const { fx, fy = fx, cx = (width - 1) / 2, cy = (height - 1) / 2 } = intrinsics;
  const maxPoints = Math.ceil(width / step) * Math.ceil(height / step);
  const positions = new Float32Array(maxPoints * 3);
  const colors = new Float32Array(maxPoints * 3);
  let count = 0;
  for (let v = 0; v < height; v += step) {
    for (let u = 0; u < width; u += step) {
      const i = v * width + u;
      const z = depth[i];
      if (!Number.isFinite(z) || z <= 0) continue;
      positions.set([((u - cx) * z) / fx, (-(v - cy) * z) / fy, -z], count * 3);
      if (rgba) {
        colors.set([rgba[i * 4] / 255, rgba[i * 4 + 1] / 255, rgba[i * 4 + 2] / 255], count * 3);
      } else {
        colors.fill(GREY, count * 3, count * 3 + 3);
      }
      count++;
    }
  }
  return { positions: positions.slice(0, count * 3), colors: colors.slice(0, count * 3), count };
}

/** Centre of the cloud's bounding box and its largest side, for framing the camera. */
export function bounds({ positions, count }: PointCloud): { centre: [number, number, number]; size: number } {
  if (count === 0) return { centre: [0, 0, 0], size: 1 };
  const min = [Infinity, Infinity, Infinity];
  const max = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < count * 3; i++) {
    const axis = i % 3;
    min[axis] = Math.min(min[axis], positions[i]);
    max[axis] = Math.max(max[axis], positions[i]);
  }
  const centre = [0, 1, 2].map((a) => (min[a] + max[a]) / 2) as [number, number, number];
  return { centre, size: Math.max(max[0] - min[0], max[1] - min[1], max[2] - min[2]) || 1 };
}

/** Maps a pointer position over an element showing the image to integer pixel coordinates. */
export function pixelAt(
  clientX: number,
  clientY: number,
  rect: { left: number; top: number; width: number; height: number },
  width: number,
  height: number,
): { x: number; y: number } | null {
  const x = Math.floor(((clientX - rect.left) / rect.width) * width);
  const y = Math.floor(((clientY - rect.top) / rect.height) * height);
  return x >= 0 && x < width && y >= 0 && y < height ? { x, y } : null;
}
