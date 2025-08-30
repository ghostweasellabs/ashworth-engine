import React from "react";
import { GitBranch, GitCommit, GitPullRequest, Plus } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";

export function GitOperationsView() {
  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Git Operations</h1>
          <p className="text-muted-foreground">
            Manage Git repository and version control
          </p>
        </div>
        <Button size="sm">
          <Plus className="h-4 w-4 mr-2" />
          New Branch
        </Button>
      </div>

      <div className="grid gap-4 md:grid-cols-3">
        <div className="rounded-lg border bg-card p-6">
          <div className="flex items-center space-x-2">
            <GitBranch className="h-4 w-4 text-blue-600" />
            <h3 className="font-semibold">Current Branch</h3>
          </div>
          <div className="mt-2">
            <div className="text-xl font-bold">main</div>
            <p className="text-xs text-muted-foreground">Up to date</p>
          </div>
        </div>

        <div className="rounded-lg border bg-card p-6">
          <div className="flex items-center space-x-2">
            <GitCommit className="h-4 w-4 text-green-600" />
            <h3 className="font-semibold">Commits</h3>
          </div>
          <div className="mt-2">
            <div className="text-xl font-bold">247</div>
            <p className="text-xs text-muted-foreground">Total commits</p>
          </div>
        </div>

        <div className="rounded-lg border bg-card p-6">
          <div className="flex items-center space-x-2">
            <GitPullRequest className="h-4 w-4 text-purple-600" />
            <h3 className="font-semibold">Changes</h3>
          </div>
          <div className="mt-2">
            <div className="text-xl font-bold">3</div>
            <p className="text-xs text-muted-foreground">Modified files</p>
          </div>
        </div>
      </div>

      <div className="rounded-lg border bg-card">
        <div className="p-4 border-b">
          <h3 className="font-semibold">Recent Commits</h3>
        </div>
        
        <div className="divide-y">
          <div className="p-4">
            <div className="flex items-start justify-between">
              <div>
                <div className="font-medium">Add command palette functionality</div>
                <div className="text-sm text-muted-foreground">
                  Implemented keyboard-accessible command palette with fuzzy search
                </div>
                <div className="flex items-center space-x-2 mt-2">
                  <Badge variant="outline">feat</Badge>
                  <span className="text-xs text-muted-foreground">2 hours ago</span>
                </div>
              </div>
              <div className="text-xs font-mono text-muted-foreground">a1b2c3d</div>
            </div>
          </div>
          
          <div className="p-4">
            <div className="flex items-start justify-between">
              <div>
                <div className="font-medium">Update shadcn/ui components</div>
                <div className="text-sm text-muted-foreground">
                  Added essential UI components for the application
                </div>
                <div className="flex items-center space-x-2 mt-2">
                  <Badge variant="outline">chore</Badge>
                  <span className="text-xs text-muted-foreground">4 hours ago</span>
                </div>
              </div>
              <div className="text-xs font-mono text-muted-foreground">e4f5g6h</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}