import type { ColorScale } from "../api/types";

// Turbo colormap stops (Mikhailov 2019), low to high, matching the server's PNGs.
export const TURBO = ["#30123b", "#4686fb", "#1be5b5", "#a4fc3c", "#fbb938", "#e4460a", "#7a0403"];

function format(v: number): string {
  const abs = Math.abs(v);
  return abs >= 100 ? v.toFixed(0) : abs >= 10 ? v.toFixed(1) : v.toFixed(2);
}

interface Props {
  src: string;
  alt: string;
  caption: string;
  scale: ColorScale;
  unit?: string;
}

/** A colour-mapped image with a legend that shows which colour means which value. */
export function Heatmap({ src, alt, caption, scale, unit = "" }: Props) {
  // A reversed scale draws the minimum red, so the bar runs red to blue from left to right.
  const stops = scale.reversed ? [...TURBO].reverse() : TURBO;
  const suffix = unit ? ` ${unit}` : "";
  return (
    <figure style={{ margin: 0 }}>
      <img src={src} alt={alt} style={{ width: "100%", display: "block" }} />
      <div
        role="img"
        aria-label={`${caption} colour scale from ${format(scale.min)} to ${format(scale.max)}${suffix}`}
        style={{
          height: 10,
          marginTop: 6,
          background: `linear-gradient(to right, ${stops.join(", ")})`,
        }}
      />
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.8rem" }}>
        <span>
          {format(scale.min)}
          {suffix}
        </span>
        <span>
          {format(scale.max)}
          {suffix}
        </span>
      </div>
      <figcaption>{caption}</figcaption>
    </figure>
  );
}
