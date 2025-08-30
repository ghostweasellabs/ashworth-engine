import React from "react";
import { Database, Upload, Search, FileText, Brain } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";

export function RAGManagementView() {
  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">RAG Management</h1>
          <p className="text-muted-foreground">
            Manage knowledge bases and document processing
          </p>
        </div>
        <Button size="sm">
          <Upload className="h-4 w-4 mr-2" />
          Upload Documents
        </Button>
      </div>

      <div className="grid gap-4 md:grid-cols-3">
        <div className="rounded-lg border bg-card p-6">
          <div className="flex items-center space-x-2">
            <Database className="h-4 w-4 text-blue-600" />
            <h3 className="font-semibold">Knowledge Bases</h3>
          </div>
          <div className="mt-2">
            <div className="text-2xl font-bold">4</div>
            <p className="text-xs text-muted-foreground">Active knowledge bases</p>
          </div>
        </div>

        <div className="rounded-lg border bg-card p-6">
          <div className="flex items-center space-x-2">
            <FileText className="h-4 w-4 text-green-600" />
            <h3 className="font-semibold">Documents</h3>
          </div>
          <div className="mt-2">
            <div className="text-2xl font-bold">1,247</div>
            <p className="text-xs text-muted-foreground">Processed documents</p>
          </div>
        </div>

        <div className="rounded-lg border bg-card p-6">
          <div className="flex items-center space-x-2">
            <Brain className="h-4 w-4 text-purple-600" />
            <h3 className="font-semibold">Embeddings</h3>
          </div>
          <div className="mt-2">
            <div className="text-2xl font-bold">15.2K</div>
            <p className="text-xs text-muted-foreground">Vector embeddings</p>
          </div>
        </div>
      </div>

      <div className="flex items-center space-x-4">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input
            placeholder="Search knowledge base..."
            className="pl-10"
          />
        </div>
        <Button variant="outline">
          Search
        </Button>
      </div>

      <div className="rounded-lg border bg-card">
        <div className="p-4 border-b">
          <h3 className="font-semibold">Knowledge Bases</h3>
        </div>
        
        <div className="divide-y">
          <div className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <div className="font-medium">Financial Regulations</div>
                <div className="text-sm text-muted-foreground">
                  IRS compliance rules and tax regulations
                </div>
                <div className="flex items-center space-x-2 mt-2">
                  <Badge variant="default">Active</Badge>
                  <span className="text-xs text-muted-foreground">342 documents</span>
                  <span className="text-xs text-muted-foreground">4.2K embeddings</span>
                </div>
              </div>
              <Button variant="outline" size="sm">
                Manage
              </Button>
            </div>
          </div>
          
          <div className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <div className="font-medium">Development Guidelines</div>
                <div className="text-sm text-muted-foreground">
                  Coding standards and best practices
                </div>
                <div className="flex items-center space-x-2 mt-2">
                  <Badge variant="default">Active</Badge>
                  <span className="text-xs text-muted-foreground">128 documents</span>
                  <span className="text-xs text-muted-foreground">1.8K embeddings</span>
                </div>
              </div>
              <Button variant="outline" size="sm">
                Manage
              </Button>
            </div>
          </div>
          
          <div className="p-4">
            <div className="flex items-center justify-between">
              <div>
                <div className="font-medium">Project Documentation</div>
                <div className="text-sm text-muted-foreground">
                  Technical specifications and requirements
                </div>
                <div className="flex items-center space-x-2 mt-2">
                  <Badge variant="secondary">Processing</Badge>
                  <span className="text-xs text-muted-foreground">89 documents</span>
                  <span className="text-xs text-muted-foreground">1.2K embeddings</span>
                </div>
              </div>
              <Button variant="outline" size="sm">
                Manage
              </Button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}