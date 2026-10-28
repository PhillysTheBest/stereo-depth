// Stand-in for the real API client until POST /v1/depth exists (Sprint 2).
// The shape mirrors what that endpoint will return, so the UI won't change when it's swapped.

export interface DepthRequest {
  left: File;
  right: File;
  engine: string;
  focalPx?: number;
  baselineM?: number;
}

export interface DepthResult {
  engine: string;
  disparityPng: string;
  depthPng: string;
  stats: { minM: number | null; maxM: number | null; medianM: number | null; validRatio: number };
  plyUrl: string | null;
  runtimeMs: number;
  metric: boolean;
}

export interface DepthClient {
  predict(req: DepthRequest): Promise<DepthResult>;
}

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

export const mockDepthClient: DepthClient = {
  async predict(req) {
    await new Promise((resolve) => setTimeout(resolve, 300));
    const metric = req.focalPx !== undefined && req.baselineM !== undefined;
    return {
      engine: req.engine,
      disparityPng: gradient(TURBO),
      depthPng: gradient([...TURBO].reverse()),
      stats: metric
        ? { minM: 1.2, maxM: 38.5, medianM: 9.7, validRatio: 0.87 }
        : { minM: null, maxM: null, medianM: null, validRatio: 0.87 },
      plyUrl: null,
      runtimeMs: 142,
      metric,
    };
  },
};
