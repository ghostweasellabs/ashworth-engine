/**
 * Production Build Script
 * 
 * Builds the application for production deployment
 */

import { build } from "vite";

console.log("🏗️  Building Combat Operations Center for production...");

try {
  await build({
    configFile: "./vite.config.ts",
    root: ".",
  });
  
  console.log("✅ Build completed successfully!");
  console.log("📦 Output directory: ./dist");
} catch (error) {
  console.error("❌ Build failed:", error);
  Deno.exit(1);
}