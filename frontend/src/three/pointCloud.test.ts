import { bounds, depthToPoints, pixelAt } from "./pointCloud";

test("back-projects the principal point straight down -z", () => {
  const depth = new Float32Array([2, 2, 2, 2, 5, 2, 2, 2, 2]);
  const cloud = depthToPoints(depth, 3, 3, { fx: 10 });
  expect(cloud.count).toBe(9);
  expect(Array.from(cloud.positions.slice(12, 15))).toEqual([0, -0, -5]);
});

test("puts pixels right of and above the centre at +x and +y", () => {
  const depth = new Float32Array([4, 4, 4, 4]);
  const [x, y] = depthToPoints(depth, 2, 2, { fx: 2, cx: 0, cy: 1 }).positions.slice(3, 5);
  expect(x).toBeCloseTo(2);
  expect(y).toBeCloseTo(2);
});

test("skips invalid depth and keeps colours aligned", () => {
  const depth = new Float32Array([NaN, 0, 3, Infinity]);
  const rgba = new Uint8ClampedArray([0, 0, 0, 255, 0, 0, 0, 255, 255, 0, 51, 255, 0, 0, 0, 255]);
  const cloud = depthToPoints(depth, 2, 2, { fx: 1 }, { rgba });
  expect(cloud.count).toBe(1);
  expect(cloud.colors).toEqual(new Float32Array([1, 0, 0.2]));
});

test("thins with step", () => {
  const cloud = depthToPoints(new Float32Array(16).fill(1), 4, 4, { fx: 1 }, { step: 2 });
  expect(cloud.count).toBe(4);
});

test("rejects mismatched sizes", () => {
  expect(() => depthToPoints(new Float32Array(3), 2, 2, { fx: 1 })).toThrow();
});

test("bounds frames the cloud", () => {
  const cloud = depthToPoints(new Float32Array([1, 1, 1, 1]), 2, 2, { fx: 1 });
  const { centre, size } = bounds(cloud);
  expect(centre[2]).toBe(-1);
  expect(size).toBeCloseTo(1);
});

test("pixelAt scales from display size to image size", () => {
  const rect = { left: 10, top: 20, width: 200, height: 100 };
  expect(pixelAt(110, 70, rect, 640, 480)).toEqual({ x: 320, y: 240 });
  expect(pixelAt(5, 70, rect, 640, 480)).toBeNull();
});
