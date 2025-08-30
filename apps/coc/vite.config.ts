import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { fileURLToPath, URL } from "node:url";

export default defineConfig({
  plugins: [react()],
  root: ".",
  publicDir: "public",
  build: {
    outDir: "dist",
    emptyOutDir: true,
    sourcemap: true,
    target: "esnext",
  },
  resolve: {
    alias: {
      "@": fileURLToPath(new URL("./src", import.meta.url)),
    },
  },
  css: {
    postcss: "./postcss.config.js",
  },
  server: {
    port: 3001,
    host: "0.0.0.0",
    open: false,
    cors: true,
  },
  preview: {
    port: 3001,
    host: "0.0.0.0",
  },
  esbuild: {
    jsx: "automatic",
    jsxImportSource: "react",
  },
});