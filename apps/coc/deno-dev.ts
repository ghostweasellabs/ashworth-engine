/**
 * Deno Development Server with TypeScript compilation
 * 
 * Alternative development setup that compiles TypeScript on-the-fly
 */

import { serve } from "https://deno.land/std@0.208.0/http/server.ts";
import { serveDir } from "https://deno.land/std@0.208.0/http/file_server.ts";
import { exists } from "https://deno.land/std@0.208.0/fs/exists.ts";

const PORT = 3001;

console.log(`🚀 Deno COC Development Server starting on port ${PORT}`);
console.log(`📱 Access the application at: http://localhost:${PORT}`);
console.log(`🔧 TypeScript compilation enabled`);

await serve(
  async (req: Request) => {
    const url = new URL(req.url);
    
    // Serve the main HTML file for the root path
    if (url.pathname === "/") {
      const htmlContent = await Deno.readTextFile("index.html");
      return new Response(htmlContent, {
        headers: { "content-type": "text/html" },
      });
    }
    
    // Handle TypeScript files
    if (url.pathname.endsWith(".ts") || url.pathname.endsWith(".tsx")) {
      const filePath = `.${url.pathname}`;
      
      if (await exists(filePath)) {
        try {
          const content = await Deno.readTextFile(filePath);
          
          // Simple TypeScript to JavaScript transformation
          let jsContent = content
            .replace(/import\s+.*?\s+from\s+["']@\/(.*?)["']/g, 'import $1 from "./$1"')
            .replace(/export\s+default\s+/g, 'export { default } from ')
            .replace(/interface\s+\w+\s*{[^}]*}/g, '') // Remove interfaces
            .replace(/:\s*\w+(\[\])?/g, '') // Remove type annotations
            .replace(/as\s+\w+/g, ''); // Remove type assertions
          
          return new Response(jsContent, {
            headers: { "content-type": "application/javascript" },
          });
        } catch (error) {
          console.error(`Error processing ${filePath}:`, error);
          return new Response(`// Error loading ${filePath}`, {
            headers: { "content-type": "application/javascript" },
          });
        }
      }
    }
    
    // Serve static files
    return serveDir(req, {
      fsRoot: ".",
      urlRoot: "",
    });
  },
  { port: PORT }
);