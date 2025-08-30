import React from "react";
import { Folder, File, Upload, Download, Plus, Search } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";

export function FileManagerView() {
  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">File Manager</h1>
          <p className="text-muted-foreground">
            Browse and manage project files
          </p>
        </div>
        <div className="flex items-center space-x-2">
          <Button variant="outline" size="sm">
            <Upload className="h-4 w-4 mr-2" />
            Upload
          </Button>
          <Button size="sm">
            <Plus className="h-4 w-4 mr-2" />
            New File
          </Button>
        </div>
      </div>

      <div className="flex items-center space-x-4">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input
            placeholder="Search files and folders..."
            className="pl-10"
          />
        </div>
        <Button variant="outline" size="sm">
          <Download className="h-4 w-4 mr-2" />
          Export
        </Button>
      </div>

      <div className="rounded-lg border bg-card">
        <div className="p-4 border-b">
          <div className="flex items-center text-sm text-muted-foreground">
            <span className="w-8"></span>
            <span className="flex-1">Name</span>
            <span className="w-24">Size</span>
            <span className="w-32">Modified</span>
            <span className="w-24">Type</span>
          </div>
        </div>
        
        <div className="divide-y">
          {/* Sample file entries */}
          <div className="p-4 hover:bg-accent/50 transition-colors cursor-pointer">
            <div className="flex items-center text-sm">
              <Folder className="h-4 w-4 mr-4 text-blue-600" />
              <span className="flex-1 font-medium">src</span>
              <span className="w-24 text-muted-foreground">-</span>
              <span className="w-32 text-muted-foreground">2 hours ago</span>
              <Badge variant="secondary" className="w-20">Folder</Badge>
            </div>
          </div>
          
          <div className="p-4 hover:bg-accent/50 transition-colors cursor-pointer">
            <div className="flex items-center text-sm">
              <Folder className="h-4 w-4 mr-4 text-blue-600" />
              <span className="flex-1 font-medium">components</span>
              <span className="w-24 text-muted-foreground">-</span>
              <span className="w-32 text-muted-foreground">1 day ago</span>
              <Badge variant="secondary" className="w-20">Folder</Badge>
            </div>
          </div>
          
          <div className="p-4 hover:bg-accent/50 transition-colors cursor-pointer">
            <div className="flex items-center text-sm">
              <File className="h-4 w-4 mr-4 text-green-600" />
              <span className="flex-1 font-medium">main.ts</span>
              <span className="w-24 text-muted-foreground">2.4 KB</span>
              <span className="w-32 text-muted-foreground">3 hours ago</span>
              <Badge variant="outline" className="w-20">TypeScript</Badge>
            </div>
          </div>
          
          <div className="p-4 hover:bg-accent/50 transition-colors cursor-pointer">
            <div className="flex items-center text-sm">
              <File className="h-4 w-4 mr-4 text-orange-600" />
              <span className="flex-1 font-medium">deno.json</span>
              <span className="w-24 text-muted-foreground">1.8 KB</span>
              <span className="w-32 text-muted-foreground">1 day ago</span>
              <Badge variant="outline" className="w-20">JSON</Badge>
            </div>
          </div>
          
          <div className="p-4 hover:bg-accent/50 transition-colors cursor-pointer">
            <div className="flex items-center text-sm">
              <File className="h-4 w-4 mr-4 text-purple-600" />
              <span className="flex-1 font-medium">README.md</span>
              <span className="w-24 text-muted-foreground">3.2 KB</span>
              <span className="w-32 text-muted-foreground">2 days ago</span>
              <Badge variant="outline" className="w-20">Markdown</Badge>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}