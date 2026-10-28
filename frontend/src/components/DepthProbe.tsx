import { useState, type MouseEvent } from "react";

import { pixelAt } from "../three/pointCloud";

interface Props {
  /** Row-major depth, NaN where unknown. */
  depth: Float32Array;
  width: number;
  height: number;
  /** What to show under the pointer: the left image or a depth heatmap of the same size. */
  imageUrl: string;
  /** "m" for metric depth; omit for relative depth. */
  unit?: string;
}

/** Shows an image and reads out the depth of the pixel under the pointer. */
export function DepthProbe({ depth, width, height, imageUrl, unit }: Props) {
  const [hover, setHover] = useState<{ x: number; y: number } | null>(null);

  function onMove(event: MouseEvent<HTMLImageElement>) {
    const rect = event.currentTarget.getBoundingClientRect();
    setHover(pixelAt(event.clientX, event.clientY, rect, width, height));
  }

  let readout = "Hover over the image to read depth";
  if (hover) {
    const z = depth[hover.y * width + hover.x];
    const value = Number.isFinite(z) ? `${z.toFixed(2)}${unit ? ` ${unit}` : " (relative)"}` : "unknown";
    readout = `(${hover.x}, ${hover.y}): ${value}`;
  }

  return (
    <figure>
      <img
        src={imageUrl}
        alt="Depth probe"
        style={{ maxWidth: "100%", cursor: "crosshair" }}
        onMouseMove={onMove}
        onMouseLeave={() => setHover(null)}
      />
      <figcaption>
        <output aria-live="polite">{readout}</output>
      </figcaption>
    </figure>
  );
}
