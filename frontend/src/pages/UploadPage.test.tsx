import { fireEvent, render, screen } from "@testing-library/react";

import type { DepthClient } from "../mocks/depthClient";
import { mockDepthClient } from "../mocks/depthClient";
import { UploadPage } from "./UploadPage";

function pick(label: string) {
  const file = new File(["x"], `${label}.png`, { type: "image/png" });
  fireEvent.change(screen.getByLabelText(label), { target: { files: [file] } });
}

test("submit is disabled until both images are chosen", () => {
  render(<UploadPage />);
  const button = screen.getByRole("button", { name: "Compute depth" });
  expect(button).toBeDisabled();
  pick("Left image");
  expect(button).toBeDisabled();
  pick("Right image");
  expect(button).toBeEnabled();
});

test("shows relative depth without calibration", async () => {
  render(<UploadPage client={mockDepthClient} />);
  pick("Left image");
  pick("Right image");
  fireEvent.click(screen.getByRole("button", { name: "Compute depth" }));
  expect(await screen.findByRole("heading", { name: "Result" })).toBeInTheDocument();
  expect(screen.getByText("Relative depth")).toBeInTheDocument();
  expect(screen.getByAltText("Disparity map")).toBeInTheDocument();
});

test("shows metric stats when focal and baseline are given", async () => {
  render(<UploadPage />);
  pick("Left image");
  pick("Right image");
  fireEvent.change(screen.getByLabelText("Focal length (px)"), { target: { value: "700" } });
  fireEvent.change(screen.getByLabelText("Baseline (m)"), { target: { value: "0.12" } });
  fireEvent.click(screen.getByRole("button", { name: "Compute depth" }));
  expect(await screen.findByText("Depth (metres)")).toBeInTheDocument();
  expect(screen.getByText(/9\.70 m/)).toBeInTheDocument();
});

test("shows the error when the client fails", async () => {
  const failing: DepthClient = { predict: () => Promise.reject(new Error("API down")) };
  render(<UploadPage client={failing} />);
  pick("Left image");
  pick("Right image");
  fireEvent.click(screen.getByRole("button", { name: "Compute depth" }));
  expect(await screen.findByRole("alert")).toHaveTextContent("API down");
});
