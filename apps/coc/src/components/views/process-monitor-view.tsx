import React from "react";
import { Activity, Square, Play, RotateCcw, AlertTriangle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";

export function ProcessMonitorView() {
  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Process Monitor</h1>
          <p className="text-muted-foreground">
            Monitor and manage system processes
          </p>
        </div>
        <Button size="sm">
          <RotateCcw className="h-4 w-4 mr-2" />
          Refresh
        </Button>
      </div>

      <div className="grid gap-4 md:grid-cols-3">
        <div className="rounded-lg border bg-card p-6">
          <div className="flex items-center space-x-2">
            <Activity className="h-4 w-4 text-green-600" />
            <h3 className="font-semibold">Running</h3>
          </div>
          <div className="mt-2">
            <div className="text-2xl font-bold">12</div>
            <p className="text-xs text-muted-foreground">Active processes</p>
          </div>
        </div>

        <div className="rounded-lg border bg-card p-6">
          <div className="flex items-center space-x-2">
            <Square className="h-4 w-4 text-gray-600" />
            <h3 className="font-semibold">Stopped</h3>
          </div>
          <div className="mt-2">
            <div className="text-2xl font-bold">3</div>
            <p className="text-xs text-muted-foreground">Inactive processes</p>
          </div>
        </div>

        <div className="rounded-lg border bg-card p-6">
          <div className="flex items-center space-x-2">
            <AlertTriangle className="h-4 w-4 text-red-600" />
            <h3 className="font-semibold">Errors</h3>
          </div>
          <div className="mt-2">
            <div className="text-2xl font-bold">1</div>
            <p className="text-xs text-muted-foreground">Failed processes</p>
          </div>
        </div>
      </div>

      <div className="rounded-lg border bg-card">
        <div className="p-4 border-b">
          <div className="flex items-center text-sm text-muted-foreground">
            <span className="w-16">PID</span>
            <span className="flex-1">Process Name</span>
            <span className="w-20">CPU</span>
            <span className="w-20">Memory</span>
            <span className="w-24">Status</span>
            <span className="w-32">Actions</span>
          </div>
        </div>
        
        <div className="divide-y">
          <div className="p-4">
            <div className="flex items-center text-sm">
              <span className="w-16 font-mono">1234</span>
              <span className="flex-1 font-medium">deno run --allow-all dev.ts</span>
              <span className="w-20 text-muted-foreground">15.2%</span>
              <span className="w-20 text-muted-foreground">128MB</span>
              <Badge variant="default" className="w-20">Running</Badge>
              <div className="w-32 flex space-x-1">
                <Button variant="outline" size="sm">
                  <Square className="h-3 w-3" />
                </Button>
                <Button variant="outline" size="sm">
                  <RotateCcw className="h-3 w-3" />
                </Button>
              </div>
            </div>
          </div>
          
          <div className="p-4">
            <div className="flex items-center text-sm">
              <span className="w-16 font-mono">5678</span>
              <span className="flex-1 font-medium">vite dev server</span>
              <span className="w-20 text-muted-foreground">8.7%</span>
              <span className="w-20 text-muted-foreground">64MB</span>
              <Badge variant="default" className="w-20">Running</Badge>
              <div className="w-32 flex space-x-1">
                <Button variant="outline" size="sm">
                  <Square className="h-3 w-3" />
                </Button>
                <Button variant="outline" size="sm">
                  <RotateCcw className="h-3 w-3" />
                </Button>
              </div>
            </div>
          </div>
          
          <div className="p-4">
            <div className="flex items-center text-sm">
              <span className="w-16 font-mono">9012</span>
              <span className="flex-1 font-medium">supabase start</span>
              <span className="w-20 text-muted-foreground">3.1%</span>
              <span className="w-20 text-muted-foreground">256MB</span>
              <Badge variant="default" className="w-20">Running</Badge>
              <div className="w-32 flex space-x-1">
                <Button variant="outline" size="sm">
                  <Square className="h-3 w-3" />
                </Button>
                <Button variant="outline" size="sm">
                  <RotateCcw className="h-3 w-3" />
                </Button>
              </div>
            </div>
          </div>
          
          <div className="p-4">
            <div className="flex items-center text-sm">
              <span className="w-16 font-mono">3456</span>
              <span className="flex-1 font-medium">python main.py</span>
              <span className="w-20 text-muted-foreground">0%</span>
              <span className="w-20 text-muted-foreground">0MB</span>
              <Badge variant="secondary" className="w-20">Stopped</Badge>
              <div className="w-32 flex space-x-1">
                <Button variant="outline" size="sm">
                  <Play className="h-3 w-3" />
                </Button>
                <Button variant="outline" size="sm">
                  <RotateCcw className="h-3 w-3" />
                </Button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}