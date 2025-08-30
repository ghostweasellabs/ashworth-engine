/**
 * Development Server Entry Point
 * 
 * Starts Vite development server with hot module replacement
 */

try {
  const { createServer } = await import("vite");
  
  const server = await createServer({
    configFile: "./vite.config.ts",
    root: ".",
    server: {
      port: 3001,
      host: "0.0.0.0",
    },
  });

  await server.listen();

  console.log("🔥 COC Development server running at http://localhost:3001");
  console.log("⚡ Hot module replacement enabled");

  server.printUrls();
} catch (error) {
  console.error("Failed to start development server:", error);
  console.log("💡 Try running: deno cache --reload dev.ts");
  Deno.exit(1);
}