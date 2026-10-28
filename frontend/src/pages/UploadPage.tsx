import { type FormEvent, useState } from "react";

import { type DepthClient, type DepthResult, mockDepthClient } from "../mocks/depthClient";
import { ResultView } from "./ResultView";

interface Props {
  client?: DepthClient;
}

function parseOptional(value: string): number | undefined {
  const n = Number(value);
  return value.trim() !== "" && Number.isFinite(n) && n > 0 ? n : undefined;
}

export function UploadPage({ client = mockDepthClient }: Props) {
  const [left, setLeft] = useState<File | null>(null);
  const [right, setRight] = useState<File | null>(null);
  const [focal, setFocal] = useState("");
  const [baseline, setBaseline] = useState("");
  const [result, setResult] = useState<DepthResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function onSubmit(e: FormEvent) {
    e.preventDefault();
    if (!left || !right) return;
    setBusy(true);
    setError(null);
    try {
      setResult(
        await client.predict({
          left,
          right,
          engine: "sgbm",
          focalPx: parseOptional(focal),
          baselineM: parseOptional(baseline),
        }),
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong");
    } finally {
      setBusy(false);
    }
  }

  return (
    <section>
      <form onSubmit={onSubmit}>
        <label>
          Left image
          <input
            type="file"
            accept="image/*"
            onChange={(e) => setLeft(e.target.files?.[0] ?? null)}
          />
        </label>
        <label>
          Right image
          <input
            type="file"
            accept="image/*"
            onChange={(e) => setRight(e.target.files?.[0] ?? null)}
          />
        </label>
        <fieldset>
          <legend>Calibration (optional, for real distances)</legend>
          <label>
            Focal length (px)
            <input inputMode="decimal" value={focal} onChange={(e) => setFocal(e.target.value)} />
          </label>
          <label>
            Baseline (m)
            <input
              inputMode="decimal"
              value={baseline}
              onChange={(e) => setBaseline(e.target.value)}
            />
          </label>
        </fieldset>
        <button type="submit" disabled={!left || !right || busy}>
          {busy ? "Computing…" : "Compute depth"}
        </button>
      </form>
      {error && <p role="alert">{error}</p>}
      {result && <ResultView result={result} />}
    </section>
  );
}
