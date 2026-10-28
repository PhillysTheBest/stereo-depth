import type { components } from "./schema";

export type Engine = components["schemas"]["EngineStatus"];
export type ColorScale = components["schemas"]["ColorScale"];

export interface DepthRequest {
  left: File;
  right: File;
  engine: string;
  focalPx?: number;
  baselineM?: number;
  includePly?: boolean;
}

/** What the UI shows for one result; the client maps the API response onto it. */
export interface DepthResult {
  engine: string;
  disparityPng: string;
  disparityScale: ColorScale;
  depthPng: string;
  depthScale: ColorScale;
  /** Metres when `metric`, else unitless relative depth (median = 1). */
  stats: { min: number | null; max: number | null; median: number | null; validRatio: number };
  plyUrl: string | null;
  runtimeMs: number;
  totalMs: number;
  metric: boolean;
}

export interface DepthClient {
  engines(): Promise<Engine[]>;
  predict(req: DepthRequest): Promise<DepthResult>;
}
