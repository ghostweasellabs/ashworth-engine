import React from "react";
import { BarChart3, Activity, Folder, Package } from "lucide-react";
import { Badge } from "@/components/ui/badge";

export function DashboardView() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Combat Operations Center</h1>
        <p className="text-muted-foreground">
          Command center for all development operations
        </p>
      </div>

      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <div className="rounded-lg border bg-card p-6">
          <div className="flex items-center space-x-2">
            <Activity className="h-4 w-4 text-green-600" />
            <h3 className="font-semibold">System Status</h3>
          </div>
          <div className="mt-2">
            <div className="text-2xl font-bold">Online</div>
            <p className="text-xs text-muted-foreground">All systems operational</p>
          </div>
        </div>

        <div className="rounded-lg border bg-card p-6">
          <div className="flex items-center space-x-2">
            <Folder className="h-4 w-4 text-blue-600" />
            <h3 className="font-semibold">Active Files</h3>
          </div>
          <div className="mt-2">
            <div className="text-2xl font-bold">1,247</div>
            <p className="text-xs text-muted-foreground">Files in workspace</p>
          </div>
        </div>

        <div className="rounded-lg border bg-card p-6">
          <div className="flex items-center space-x-2">
            <Package className="h-4 w-4 text-purple-600" />
            <h3 className="font-semibold">Packages</h3>
          </div>
          <div className="mt-2">
            <div className="text-2xl font-bold">42</div>
            <p className="text-xs text-muted-foreground">Dependencies installed</p>
          </div>
        </div>

        <div className="rounded-lg border bg-card p-6">
          <div className="flex items-center space-x-2">
            <BarChart3 className="h-4 w-4 text-orange-600" />
            <h3 className="font-semibold">Performance</h3>
          </div>
          <div className="mt-2">
            <div className="text-2xl font-bold">98%</div>
            <p className="text-xs text-muted-foreground">System efficiency</p>
          </div>
        </div>
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        <div className="rounded-lg border bg-card p-6">
          <h3 className="font-semibold mb-4">Recent Activity</h3>
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-sm">File system scan completed</span>
              <Badge variant="secondary">2m ago</Badge>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm">Package dependencies updated</span>
              <Badge variant="secondary">5m ago</Badge>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm">Git repository synchronized</span>
              <Badge variant="secondary">12m ago</Badge>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm">Agent workflow executed</span>
              <Badge variant="secondary">18m ago</Badge>
            </div>
          </div>
        </div>

        <div className="rounded-lg border bg-card p-6">
          <h3 className="font-semibold mb-4">Quick Actions</h3>
          <div className="space-y-2">
            <button className="w-full text-left p-2 rounded hover:bg-accent transition-colors">
              <div className="font-medium">Open Terminal</div>
              <div className="text-sm text-muted-foreground">Launch integrated terminal</div>
            </button>
            <button className="w-full text-left p-2 rounded hover:bg-accent transition-colors">
              <div className="font-medium">File Manager</div>
              <div className="text-sm text-muted-foreground">Browse project files</div>
            </button>
            <button className="w-full text-left p-2 rounded hover:bg-accent transition-colors">
              <div className="font-medium">Process Monitor</div>
              <div className="text-sm text-muted-foreground">View running processes</div>
            </button>
            <button className="w-full text-left p-2 rounded hover:bg-accent transition-colors">
              <div className="font-medium">Agent Chat</div>
              <div className="text-sm text-muted-foreground">Interact with AI agents</div>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}