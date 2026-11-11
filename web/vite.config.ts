import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

// In development the page comes from Vite and the API from `rpa serve`.
export default defineConfig({
  plugins: [react()],
  server: { proxy: { "/api": "http://127.0.0.1:8000" } },
  build: { chunkSizeWarningLimit: 1500 },
  test: { environment: "node", include: ["src/**/*.test.ts"] },
});
