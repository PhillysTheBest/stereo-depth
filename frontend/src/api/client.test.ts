import { buildDepthForm, createDepthClient } from "./client";

const apiResult = {
  engine: "raft_stereo",
  width: 4,
  height: 3,
  scale: 1,
  unit: "m",
  runtime_ms: 12,
  total_ms: 20,
  disparity_png: "data:image/png;base64,AA==",
  disparity_scale: { min: 1, max: 9, colormap: "turbo", reversed: false },
  depth_png: "data:image/png;base64,AA==",
  depth_scale: { min: 0.5, max: 4, colormap: "turbo", reversed: true },
  depth_stats: { min: 0.5, max: 4, mean: 2, median: 1.8, valid_fraction: 0.9 },
  ply: null,
};

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

test("builds the multipart form the API expects", () => {
  const left = new File(["l"], "l.png", { type: "image/png" });
  const right = new File(["r"], "r.png", { type: "image/png" });
  const form = buildDepthForm({ left, right, engine: "raft_stereo", focalPx: 700, baselineM: 0.1 });
  expect(form.get("engine")).toBe("raft_stereo");
  expect(form.get("focal_px")).toBe("700");
  expect(form.get("baseline_m")).toBe("0.1");
  expect(form.get("include_ply")).toBeNull();
  expect((form.get("left") as File).name).toBe("l.png");
  expect(buildDepthForm({ left, right, engine: "sgbm", includePly: true }).get("include_ply")).toBe(
    "true",
  );
});

test("posts to /v1/depth and maps the response", async () => {
  const fetchMock = vi.fn(async () => json(apiResult));
  const client = createDepthClient("http://api.test", fetchMock as unknown as typeof fetch);
  const file = new File(["x"], "x.png", { type: "image/png" });

  const result = await client.predict({ left: file, right: file, engine: "raft_stereo" });

  const req = (fetchMock.mock.calls[0] as unknown as [Request])[0];
  expect(req.url).toBe("http://api.test/v1/depth");
  expect(req.method).toBe("POST");
  expect(result).toMatchObject({
    metric: true,
    runtimeMs: 12,
    stats: { median: 1.8, validRatio: 0.9 },
    depthScale: { reversed: true },
  });
});

test("surfaces the API's error detail", async () => {
  const fetchMock = vi.fn(async () => json({ detail: "images differ in size" }, 400));
  const client = createDepthClient("http://api.test", fetchMock as unknown as typeof fetch);
  const file = new File(["x"], "x.png");
  await expect(client.predict({ left: file, right: file, engine: "sgbm" })).rejects.toThrow(
    "images differ in size",
  );
});

test("lists engines", async () => {
  const engines = [
    { name: "sgbm", label: "SGBM", requires_gpu: false, available: true, loaded: true },
  ];
  const client = createDepthClient("http://api.test", (async () =>
    json(engines)) as unknown as typeof fetch);
  expect(await client.engines()).toEqual(engines);
});
