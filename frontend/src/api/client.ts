import createClient from "openapi-fetch";

import type { paths } from "./schema";
import type { DepthClient, DepthRequest, DepthResult } from "./types";

type ApiDepthResult = paths["/v1/depth"]["post"]["responses"][200]["content"]["application/json"];

/** Base URL of the API. Empty means same origin (the Vite dev server proxies /v1). */
export const API_URL: string = import.meta.env.VITE_API_URL ?? "";

function errorMessage(error: unknown, status: number): string {
  const detail = (error as { detail?: unknown } | undefined)?.detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return detail.map((d: { msg?: string }) => d.msg ?? JSON.stringify(d)).join("; ");
  }
  return `Request failed (HTTP ${status})`;
}

export function toDepthResult(body: ApiDepthResult): DepthResult {
  const s = body.depth_stats;
  return {
    engine: body.engine,
    disparityPng: body.disparity_png,
    disparityScale: body.disparity_scale,
    depthPng: body.depth_png,
    depthScale: body.depth_scale,
    stats: { min: s.min, max: s.max, median: s.median, validRatio: s.valid_fraction },
    plyUrl: body.ply ?? null,
    runtimeMs: body.runtime_ms,
    totalMs: body.total_ms,
    metric: body.unit === "m",
  };
}

/** The multipart body POST /v1/depth expects. */
export function buildDepthForm(req: DepthRequest): FormData {
  const form = new FormData();
  form.append("left", req.left);
  form.append("right", req.right);
  form.append("engine", req.engine);
  if (req.focalPx !== undefined) form.append("focal_px", String(req.focalPx));
  if (req.baselineM !== undefined) form.append("baseline_m", String(req.baselineM));
  if (req.includePly) form.append("include_ply", "true");
  return form;
}

export function createDepthClient(
  baseUrl: string = API_URL,
  fetchImpl?: typeof fetch,
): DepthClient {
  const api = createClient<paths>({ baseUrl, fetch: fetchImpl });

  return {
    async engines() {
      const { data, error, response } = await api.GET("/v1/engines");
      if (!data) throw new Error(errorMessage(error, response.status));
      return data;
    },

    async predict(req: DepthRequest) {
      const form = buildDepthForm(req);
      const { data, error, response } = await api.POST("/v1/depth", {
        // The generated body type describes the fields; FormData carries them as multipart.
        body: form as unknown as paths["/v1/depth"]["post"]["requestBody"]["content"]["multipart/form-data"],
        bodySerializer: (body) => body as unknown as FormData,
      });
      if (!data) throw new Error(errorMessage(error, response.status));
      return toDepthResult(data);
    },
  };
}
