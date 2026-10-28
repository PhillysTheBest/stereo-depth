import type { Engine } from "../api/types";

interface Props {
  engines: Engine[];
  value: string;
  onChange: (name: string) => void;
  disabled?: boolean;
}

/** Radio group of the engines the API offers. Engines without weights are shown but disabled. */
export function ModelPicker({ engines, value, onChange, disabled = false }: Props) {
  return (
    <fieldset disabled={disabled}>
      <legend>Engine</legend>
      {engines.map((engine) => (
        <label key={engine.name} style={{ display: "block" }}>
          <input
            type="radio"
            name="engine"
            value={engine.name}
            checked={value === engine.name}
            disabled={!engine.available}
            onChange={() => onChange(engine.name)}
          />
          {engine.label}
          {!engine.available && " (not ready on this server)"}
          {engine.available && !engine.loaded && " (first run loads the model)"}
        </label>
      ))}
    </fieldset>
  );
}

/** The engine to select first: the preferred one if usable, else the first available. */
export function defaultEngine(engines: Engine[], preferred = "raft_stereo"): string {
  const usable = engines.filter((e) => e.available);
  return (usable.find((e) => e.name === preferred) ?? usable[0])?.name ?? "";
}
