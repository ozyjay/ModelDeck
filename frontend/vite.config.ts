import { fileURLToPath, URL } from "node:url";

import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

const defaultOutput = fileURLToPath(new URL("../backend/modeldeck/api/static", import.meta.url));

export default defineConfig({
  plugins: [react()],
  build: {
    outDir: process.env.MODELDECK_FRONTEND_OUT_DIR || defaultOutput,
    emptyOutDir: true,
  },
  server: {
    host: "127.0.0.1",
    port: 5173,
    proxy: {
      "/api": "http://127.0.0.1:3600",
    },
  },
  test: {
    environment: "jsdom",
    setupFiles: ["./src/test/setup.ts"],
    restoreMocks: true,
  },
});
