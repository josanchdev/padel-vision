import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// A static viewer (ADR-0017): no API to proxy, no server of its own. The points
// it shows are plain files under public/points/, written by
// scripts/export_points.py. A relative base lets the build run from any folder
// or static host.
export default defineConfig({
  plugins: [react()],
  base: "./",
});
