/**
 * Combat Operations Center (COC) - Main Entry Point
 * 
 * A modern web-based command center that provides all CLI functionality
 * through an intuitive, powerful interface built with Deno, Vite, and IBM Carbon.
 */

import { serve } from "https://deno.land/std@0.208.0/http/server.ts";
import { serveDir } from "https://deno.land/std@0.208.0/http/file_server.ts";

const PORT = 3001;

console.log(`🚀 Combat Operations Center starting on port ${PORT}`);
console.log(`📱 Access the application at: http://localhost:${PORT}`);

// In development, we'll use Vite's dev server
// This main.ts serves as the production entry point
await serve(
  (req: Request) => {
    return serveDir(req, {
      fsRoot: "dist",
      urlRoot: "",
    });
  },
  { port: PORT }
);