import type { DepthResult } from "../mocks/depthClient";

function metres(v: number | null): string {
  return v === null ? "n/a" : `${v.toFixed(2)} m`;
}

export function ResultView({ result }: { result: DepthResult }) {
  const { stats } = result;
  return (
    <section aria-label="Result">
      <h2>Result</h2>
      <div style={{ display: "grid", gap: "1rem", gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))" }}>
        <figure>
          <img src={result.disparityPng} alt="Disparity map" style={{ width: "100%" }} />
          <figcaption>Disparity</figcaption>
        </figure>
        <figure>
          <img src={result.depthPng} alt="Depth map" style={{ width: "100%" }} />
          <figcaption>{result.metric ? "Depth (metres)" : "Relative depth"}</figcaption>
        </figure>
      </div>
      <dl>
        <dt>Engine</dt>
        <dd>{result.engine}</dd>
        <dt>Runtime</dt>
        <dd>{result.runtimeMs.toFixed(0)} ms</dd>
        <dt>Valid pixels</dt>
        <dd>{(stats.validRatio * 100).toFixed(0)}%</dd>
        {result.metric && (
          <>
            <dt>Nearest / median / farthest</dt>
            <dd>
              {metres(stats.minM)} / {metres(stats.medianM)} / {metres(stats.maxM)}
            </dd>
          </>
        )}
      </dl>
      {result.plyUrl && (
        <a href={result.plyUrl} download>
          Download point cloud (PLY)
        </a>
      )}
    </section>
  );
}
