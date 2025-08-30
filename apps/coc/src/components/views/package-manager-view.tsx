import React from "react";
import { Package, Download, Trash2, RefreshCw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";

export function PackageManagerView() {
  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Package Manager</h1>
          <p className="text-muted-foreground">
            Manage Deno and Python packages
          </p>
        </div>
        <Button size="sm">
          <RefreshCw className="h-4 w-4 mr-2" />
          Refresh
        </Button>
      </div>

      <div className="flex items-center space-x-4">
        <Input
          placeholder="Search packages..."
          className="flex-1"
        />
        <Button variant="outline">
          <Download className="h-4 w-4 mr-2" />
          Install Package
        </Button>
      </div>

      <div className="rounded-lg border bg-card">
        <div className="p-4 border-b">
          <h3 className="font-semibold">Installed Packages</h3>
        </div>
        
        <div className="divide-y">
          <div className="p-4 flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <Package className="h-4 w-4 text-blue-600" />
              <div>
                <div className="font-medium">react</div>
                <div className="text-sm text-muted-foreground">v18.3.1</div>
              </div>
            </div>
            <div className="flex items-center space-x-2">
              <Badge variant="secondary">Deno</Badge>
              <Button variant="outline" size="sm">
                <Trash2 className="h-3 w-3" />
              </Button>
            </div>
          </div>
          
          <div className="p-4 flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <Package className="h-4 w-4 text-green-600" />
              <div>
                <div className="font-medium">tailwindcss</div>
                <div className="text-sm text-muted-foreground">v3.4.0</div>
              </div>
            </div>
            <div className="flex items-center space-x-2">
              <Badge variant="secondary">Deno</Badge>
              <Button variant="outline" size="sm">
                <Trash2 className="h-3 w-3" />
              </Button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}