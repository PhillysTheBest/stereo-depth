import { fireEvent, render, screen } from "@testing-library/react";

import type { DepthClient } from "../api/types";
import { MOCK_ENGINES, mockDepthClient } from "../mocks/depthClient";
import { UploadPage } from "./UploadPage";

function pick(label: string) {
  const file = new File(["x"], `${label}.png`, { type: "image/png" });
  fireEvent.change(screen.getByLabelText(label), { target: { files: [file] } });
}

test("submit is disabled until both images are chosen", async () => {
  render(<UploadPage client={mockDepthClient} />);
  await screen.findByRole("radio", { name: /RAFT-Stereo/ });
  const button = screen.getByRole("button", { name: "Compute depth" });
  expect(button).toBeDisabled();
  pick("Left image");
  expect(button).toBeDisabled();
  pick("Right image");
  expect(button).toBeEnabled();
});

test("picks RAFT-Stereo by default and sends the chosen engine", async () => {
  const predict = vi.fn(mockDepthClient.predict);
  render(<UploadPage client={{ ...mockDepthClient, predict }} />);
  expect(await screen.findByRole("radio", { name: /RAFT-Stereo/ })).toBeChecked();
  fireEvent.click(screen.getByRole("radio", { name: /SGBM/ }));
  pick("Left image");
  pick("Right image");
  fireEvent.click(screen.getByRole("button", { name: "Compute depth" }));
  expect(await screen.findByRole("heading", { name: "Result" })).toBeInTheDocument();
  expect(predict).toHaveBeenCalledWith(expect.objectContaining({ engine: "sgbm" }));
});

test("shows relative depth without calibration", async () => {
  render(<UploadPage client={mockDepthClient} />);
  await screen.findByRole("radio", { name: /RAFT-Stereo/ });
  pick("Left image");
  pick("Right image");
  fireEvent.click(screen.getByRole("button", { name: "Compute depth" }));
  expect(screen.getByRole("status")).toHaveTextContent(/loading the model first/);
  expect(await screen.findByRole("heading", { name: "Result" })).toBeInTheDocument();
  expect(screen.getByText("Relative depth")).toBeInTheDocument();
  expect(screen.getByAltText("Disparity map")).toBeInTheDocument();
  expect(
    screen.getByRole("img", { name: /Disparity colour scale from 2.00 to 96.0 px/ }),
  ).toBeVisible();
});

test("shows metric stats when focal and baseline are given", async () => {
  render(<UploadPage client={mockDepthClient} />);
  await screen.findByRole("radio", { name: /RAFT-Stereo/ });
  pick("Left image");
  pick("Right image");
  fireEvent.change(screen.getByLabelText("Focal length (px)"), { target: { value: "700" } });
  fireEvent.change(screen.getByLabelText("Baseline (m)"), { target: { value: "0.12" } });
  fireEvent.click(screen.getByRole("button", { name: "Compute depth" }));
  expect(await screen.findByText("Depth (metres)")).toBeInTheDocument();
  expect(screen.getByText(/9\.70 m/)).toBeInTheDocument();
});

test("shows the error when the client fails", async () => {
  const failing: DepthClient = {
    engines: () => Promise.resolve(MOCK_ENGINES),
    predict: () => Promise.reject(new Error("API down")),
  };
  render(<UploadPage client={failing} />);
  await screen.findByRole("radio", { name: /RAFT-Stereo/ });
  pick("Left image");
  pick("Right image");
  fireEvent.click(screen.getByRole("button", { name: "Compute depth" }));
  expect(await screen.findByRole("alert")).toHaveTextContent("API down");
});

test("explains when the API is unreachable", async () => {
  const offline: DepthClient = {
    engines: () => Promise.reject(new Error("Failed to fetch")),
    predict: () => Promise.reject(new Error("unused")),
  };
  render(<UploadPage client={offline} />);
  expect(await screen.findByRole("alert")).toHaveTextContent(
    "Could not reach the depth API: Failed to fetch",
  );
  expect(screen.getByRole("button", { name: "Compute depth" })).toBeDisabled();
});

test("disables engines that are not ready", async () => {
  const engines = [{ ...MOCK_ENGINES[1], available: false }, MOCK_ENGINES[0]];
  render(<UploadPage client={{ ...mockDepthClient, engines: () => Promise.resolve(engines) }} />);
  expect(await screen.findByRole("radio", { name: /RAFT-Stereo/ })).toBeDisabled();
  expect(screen.getByRole("radio", { name: /SGBM/ })).toBeChecked();
});
