import { type FormEvent, useEffect, useState } from "react";

import { createDepthClient } from "../api/client";
import type { DepthClient, DepthResult, Engine } from "../api/types";
import { defaultEngine, ModelPicker } from "../components/ModelPicker";
import { ResultView } from "./ResultView";

interface Props {
  client?: DepthClient;
}

const defaultClient = createDepthClient();

function parseOptional(value: string): number | undefined {
  const n = Number(value);
  return value.trim() !== "" && Number.isFinite(n) && n > 0 ? n : undefined;
}

function message(err: unknown): string {
  return err instanceof Error ? err.message : "Something went wrong";
}

export function UploadPage({ client = defaultClient }: Props) {
  const [engines, setEngines] = useState<Engine[] | null>(null);
  const [enginesError, setEnginesError] = useState<string | null>(null);
  const [engine, setEngine] = useState("");
  const [left, setLeft] = useState<File | null>(null);
  const [right, setRight] = useState<File | null>(null);
  const [focal, setFocal] = useState("");
  const [baseline, setBaseline] = useState("");
  const [includePly, setIncludePly] = useState(false);
  const [result, setResult] = useState<DepthResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    let cancelled = false;
    client.engines().then(
      (list) => {
        if (cancelled) return;
        setEngines(list);
        setEngine((current) => current || defaultEngine(list));
      },
      (err: unknown) => {
        if (!cancelled) setEnginesError(`Could not reach the depth API: ${message(err)}`);
      },
    );
    return () => {
      cancelled = true;
    };
  }, [client]);

  const chosen = engines?.find((e) => e.name === engine);

  async function onSubmit(e: FormEvent) {
    e.preventDefault();
    if (!left || !right || !engine) return;
    setBusy(true);
    setError(null);
    try {
      setResult(
        await client.predict({
          left,
          right,
          engine,
          focalPx: parseOptional(focal),
          baselineM: parseOptional(baseline),
          includePly,
        }),
      );
    } catch (err) {
      setError(message(err));
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
        {engines === null && !enginesError && <p role="status">Loading engines…</p>}
        {enginesError && <p role="alert">{enginesError}</p>}
        {engines && (
          <ModelPicker engines={engines} value={engine} onChange={setEngine} disabled={busy} />
        )}
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
        <label>
          <input
            type="checkbox"
            checked={includePly}
            onChange={(e) => setIncludePly(e.target.checked)}
          />
          Include point cloud (PLY)
        </label>
        <button type="submit" disabled={!left || !right || !engine || busy}>
          {busy ? "Computing…" : "Compute depth"}
        </button>
      </form>
      {busy && (
        <p role="status">
          Running {chosen?.label ?? engine}
          {chosen && !chosen.loaded ? ", loading the model first" : ""}. Networks on a CPU server
          can take tens of seconds.
        </p>
      )}
      {error && <p role="alert">{error}</p>}
      {result && !busy && <ResultView result={result} />}
    </section>
  );
}
