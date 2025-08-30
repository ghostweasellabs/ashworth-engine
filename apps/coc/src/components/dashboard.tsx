import React from "react";
import { useTheme } from "../contexts/theme-context";

export function Dashboard() {
  const { theme, actualTheme } = useTheme();

  return (
    <div className="space-y-8">
      <div className="space-y-2">
        <h1 className="text-4xl font-bold tracking-tight bg-gradient-to-r from-foreground via-foreground/80 to-foreground/60 bg-clip-text text-transparent">
          Dashboard
        </h1>
        <p className="text-muted-foreground/80 text-lg">Welcome to your Combat Operations Center</p>
      </div>

      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
        {/* System Status Card */}
        <div className="glass rounded-2xl p-6 hover:scale-[1.02] transition-all duration-200 shadow-xl">
          <div className="flex items-center justify-between pb-4">
            <h3 className="text-sm font-medium text-muted-foreground">System Status</h3>
            <div className="w-3 h-3 rounded-full bg-green-500 animate-pulse shadow-lg shadow-green-500/50"></div>
          </div>
          <div className="space-y-2">
            <div className="text-3xl font-bold bg-gradient-to-r from-green-500 to-emerald-500 bg-clip-text text-transparent">
              Online
            </div>
            <p className="text-xs text-muted-foreground/70">All systems operational</p>
          </div>
        </div>

        {/* Theme Status Card */}
        <div className="glass rounded-2xl p-6 hover:scale-[1.02] transition-all duration-200 shadow-xl">
          <div className="flex items-center justify-between pb-4">
            <h3 className="text-sm font-medium text-muted-foreground">Theme</h3>
            <div className="w-3 h-3 rounded-full bg-primary animate-pulse shadow-lg shadow-primary/50"></div>
          </div>
          <div className="space-y-2">
            <div className="text-3xl font-bold bg-gradient-to-r from-primary to-primary/70 bg-clip-text text-transparent capitalize">
              {actualTheme}
            </div>
            <p className="text-xs text-muted-foreground/70">
              {theme === "system" ? "Following system preference" : `Manually set to ${theme}`}
            </p>
          </div>
        </div>

        {/* Typography Card */}
        <div className="glass rounded-2xl p-6 hover:scale-[1.02] transition-all duration-200 shadow-xl">
          <div className="flex items-center justify-between pb-4">
            <h3 className="text-sm font-medium text-muted-foreground">Typography</h3>
            <div className="w-3 h-3 rounded-full bg-purple-500 animate-pulse shadow-lg shadow-purple-500/50"></div>
          </div>
          <div className="space-y-2">
            <div className="text-3xl font-bold bg-gradient-to-r from-purple-500 to-pink-500 bg-clip-text text-transparent">
              IBM Plex
            </div>
            <p className="text-xs text-muted-foreground/70 font-mono">Professional fonts loaded</p>
          </div>
        </div>
      </div>

      <div className="glass rounded-2xl p-8 shadow-xl">
        <h3 className="text-xl font-semibold mb-6 bg-gradient-to-r from-foreground to-foreground/70 bg-clip-text text-transparent">
          Available Features
        </h3>
        <div className="grid gap-6 md:grid-cols-2">
          <div className="space-y-4">
            <h4 className="font-medium text-muted-foreground">Core Operations</h4>
            <div className="space-y-3">
              {['File Management Interface', 'Process Monitoring Dashboard', 'Package Management GUI', 'Git Operations Interface'].map((feature, i) => (
                <div key={i} className="flex items-center space-x-3 p-3 rounded-xl glass hover:bg-accent/20 transition-all duration-200">
                  <div className="w-2 h-2 rounded-full bg-primary/60"></div>
                  <span className="text-sm text-muted-foreground">{feature}</span>
                </div>
              ))}
            </div>
          </div>
          <div className="space-y-4">
            <h4 className="font-medium text-muted-foreground">AI & Intelligence</h4>
            <div className="space-y-3">
              {['Agent Chat Interface', 'RAG Management Tools', 'Real-time Analytics', 'Smart Notifications'].map((feature, i) => (
                <div key={i} className="flex items-center space-x-3 p-3 rounded-xl glass hover:bg-accent/20 transition-all duration-200">
                  <div className="w-2 h-2 rounded-full bg-purple-500/60"></div>
                  <span className="text-sm text-muted-foreground">{feature}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      <div className="glass rounded-2xl p-8 shadow-xl">
        <h3 className="text-xl font-semibold mb-6 bg-gradient-to-r from-foreground to-foreground/70 bg-clip-text text-transparent">
          Technology Stack
        </h3>
        <div className="flex flex-wrap gap-3">
          {[
            "Deno 2.0+",
            "Vite",
            "React 18",
            "TypeScript",
            "Tailwind CSS",
            "shadcn/ui"
          ].map((tech) => (
            <span
              key={tech}
              className="inline-flex items-center rounded-xl glass px-4 py-2 text-sm font-medium hover:scale-105 transition-all duration-200"
            >
              {tech}
            </span>
          ))}
        </div>
      </div>
    </div>
  );
}