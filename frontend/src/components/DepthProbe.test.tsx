import { fireEvent, render, screen } from "@testing-library/react";

import { DepthProbe } from "./DepthProbe";

function setup(unit?: string) {
  const depth = new Float32Array([1, 2, NaN, 4.5]);
  render(<DepthProbe depth={depth} width={2} height={2} imageUrl="data:," unit={unit} />);
  const img = screen.getByAltText("Depth probe");
  img.getBoundingClientRect = () => ({ left: 0, top: 0, width: 100, height: 100 }) as DOMRect;
  return img;
}

test("reads metric depth under the pointer", () => {
  const img = setup("m");
  fireEvent.mouseMove(img, { clientX: 75, clientY: 75 });
  expect(screen.getByRole("status")).toHaveTextContent("(1, 1): 4.50 m");
});

test("labels relative depth and unknown pixels", () => {
  const img = setup();
  fireEvent.mouseMove(img, { clientX: 75, clientY: 25 });
  expect(screen.getByRole("status")).toHaveTextContent("(1, 0): 2.00 (relative)");
  fireEvent.mouseMove(img, { clientX: 25, clientY: 75 });
  expect(screen.getByRole("status")).toHaveTextContent("unknown");
  fireEvent.mouseLeave(img);
  expect(screen.getByRole("status")).toHaveTextContent("Hover over the image");
});
