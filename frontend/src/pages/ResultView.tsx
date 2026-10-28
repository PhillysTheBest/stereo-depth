import type { DepthResult } from "../api/types";
import { Heatmap } from "../components/Heatmap";

function value(v: number | null, metric: boolean): string {
  if (v === null) return "n/a";
  return metric ? `${v.toFixed(2)} m` : `${v.toFixed(2)}×`;
}

export function ResultView({ result }: { result: DepthResult }) {
  const { stats, metric } = result;
  return (
    <section aria-label="Result">
      <h2>Result</h2>
      <div
        style={{
          display: "grid",
          gap: "1rem",
          gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))",
        }}
      >
        <Heatmap
          src={result.disparityPng}
          alt="Disparity map"
          caption="Disparity"
          scale={result.disparityScale}
          unit="px"
        />
        <Heatmap
          src={result.depthPng}
          alt="Depth map"
          caption={metric ? "Depth (metres)" : "Relative depth"}
          scale={result.depthScale}
          unit={metric ? "m" : ""}
        />
      </div>
      <dl>
        <dt>Engine</dt>
        <dd>{result.engine}</dd>
        <dt>Runtime</dt>
        <dd>
          {result.runtimeMs.toFixed(0)} ms in the engine, {result.totalMs.toFixed(0)} ms total
        </dd>
        <dt>Valid pixels</dt>
        <dd>{(stats.validRatio * 100).toFixed(0)}%</dd>
        <dt>Nearest / median / farthest</dt>
        <dd>
          {value(stats.min, metric)} / {value(stats.median, metric)} / {value(stats.max, metric)}
        </dd>
      </dl>
      {!metric && (
        <p>Relative depth: 1× is the median distance. Add focal length and baseline for metres.</p>
      )}
      {result.plyUrl && (
        <a href={result.plyUrl} download={`${result.engine}.ply`}>
          Download point cloud (PLY)
        </a>
      )}
    </section>
  );
}
