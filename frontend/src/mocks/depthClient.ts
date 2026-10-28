// Offline stand-in for the API, used by tests and by `VITE_USE_MOCK=1 pnpm dev`.
import type { DepthClient, Engine } from "../api/types";

export type { DepthClient, DepthRequest, DepthResult } from "../api/types";

function gradient(stops: string[]): string {
  const offsets = stops
    .map((c, i) => `<stop offset="${(i / (stops.length - 1)).toFixed(2)}" stop-color="${c}"/>`)
    .join("");
  const svg =
    `<svg xmlns="http://www.w3.org/2000/svg" width="320" height="96">` +
    `<defs><linearGradient id="g">${offsets}</linearGradient></defs>` +
    `<rect width="320" height="96" fill="url(#g)"/></svg>`;
  return `data:image/svg+xml;utf8,${encodeURIComponent(svg)}`;
}

const TURBO = ["#30123b", "#4686fb", "#1be5b5", "#a4fc3c", "#fbb938", "#e4460a", "#7a0403"];

export const MOCK_ENGINES: Engine[] = [
  {
    name: "sgbm",
    label: "OpenCV SGBM (classical)",
    requires_gpu: false,
    available: true,
    loaded: true,
  },
  {
    name: "raft_stereo",
    label: "RAFT-Stereo (pretrained network)",
    requires_gpu: false,
    available: true,
    loaded: false,
  },
];

export const mockDepthClient: DepthClient = {
  async engines() {
    return MOCK_ENGINES;
  },
  async predict(req) {
    await new Promise((resolve) => setTimeout(resolve, 300));
    const metric = req.focalPx !== undefined && req.baselineM !== undefined;
    return {
      engine: req.engine,
      disparityPng: gradient(TURBO),
      disparityScale: { min: 2, max: 96, colormap: "turbo", reversed: false },
      depthPng: gradient([...TURBO].reverse()),
      depthScale: metric
        ? { min: 1.2, max: 38.5, colormap: "turbo", reversed: true }
        : { min: 0.3, max: 3.1, colormap: "turbo", reversed: true },
      stats: metric
        ? { min: 1.2, max: 38.5, median: 9.7, validRatio: 0.87 }
        : { min: 0.3, max: 3.1, median: 1, validRatio: 0.87 },
      plyUrl: null,
      runtimeMs: 142,
      totalMs: 180,
      metric,
    };
  },
};
