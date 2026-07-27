import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// The build output is served by FastAPI as static files (ADR-0011), so it lands
// in the api package's static dir. In dev, the API's routes are proxied to the
// running FastAPI server so fetch("/matches") etc. work without CORS.
export default defineConfig({
  plugins: [react()],
  build: {
    outDir: "../api/src/padel_api/static",
    emptyOutDir: true,
  },
  server: {
    proxy: {
      "/matches": "http://localhost:8000",
      "/health": "http://localhost:8000",
    },
  },
});
