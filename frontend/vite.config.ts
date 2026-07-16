import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// In dev the Vite server proxies /api to the FastAPI backend.
// In production FastAPI serves the built dist/ itself (single container).
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: { "/api": "http://localhost:8000" },
  },
});
