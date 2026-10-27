/// <reference types="vitest/config" />
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react()],
  server: { proxy: { "/v1": "http://localhost:7860" } },
  test: { environment: "jsdom", globals: true, setupFiles: ["./src/setupTests.ts"] },
});
