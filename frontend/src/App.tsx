import { mockDepthClient } from "./mocks/depthClient";
import { UploadPage } from "./pages/UploadPage";

// VITE_USE_MOCK=1 runs the UI without a backend.
const client = import.meta.env.VITE_USE_MOCK === "1" ? mockDepthClient : undefined;

export function App() {
  return (
    <main>
      <h1>Stereo Depth</h1>
      <p>Upload a left and right image to get disparity, depth and a point cloud.</p>
      <UploadPage client={client} />
    </main>
  );
}
