/**
 * Production Preview Server
 * 
 * Serves the built application for production testing
 */

import { preview } from "vite";

console.log("🔍 Starting production preview server...");

const server = await preview({
  configFile: "./vite.config.ts",
  root: ".",
  preview: {
    port: 3001,
    host: "0.0.0.0",
  },
});

console.log("🚀 Preview server running at http://localhost:3001");
server.printUrls();