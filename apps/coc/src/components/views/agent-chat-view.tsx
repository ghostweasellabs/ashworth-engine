import React from "react";
import { MessageSquare, Send, Bot, User } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";

export function AgentChatView() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Agent Chat</h1>
        <p className="text-muted-foreground">
          Interact with AI agents and monitor workflows
        </p>
      </div>

      <div className="grid gap-4 md:grid-cols-4">
        <div className="rounded-lg border bg-card p-4">
          <div className="flex items-center space-x-2 mb-2">
            <Bot className="h-4 w-4 text-blue-600" />
            <h3 className="font-semibold">Data Processor</h3>
          </div>
          <Badge variant="default" className="mb-2">Online</Badge>
          <p className="text-xs text-muted-foreground">Financial data analysis</p>
        </div>

        <div className="rounded-lg border bg-card p-4">
          <div className="flex items-center space-x-2 mb-2">
            <Bot className="h-4 w-4 text-green-600" />
            <h3 className="font-semibold">Report Generator</h3>
          </div>
          <Badge variant="default" className="mb-2">Online</Badge>
          <p className="text-xs text-muted-foreground">Executive reporting</p>
        </div>

        <div className="rounded-lg border bg-card p-4">
          <div className="flex items-center space-x-2 mb-2">
            <Bot className="h-4 w-4 text-purple-600" />
            <h3 className="font-semibold">Code Assistant</h3>
          </div>
          <Badge variant="secondary" className="mb-2">Idle</Badge>
          <p className="text-xs text-muted-foreground">Development support</p>
        </div>

        <div className="rounded-lg border bg-card p-4">
          <div className="flex items-center space-x-2 mb-2">
            <Bot className="h-4 w-4 text-orange-600" />
            <h3 className="font-semibold">System Monitor</h3>
          </div>
          <Badge variant="default" className="mb-2">Online</Badge>
          <p className="text-xs text-muted-foreground">Infrastructure monitoring</p>
        </div>
      </div>

      <div className="rounded-lg border bg-card flex flex-col h-96">
        <div className="p-4 border-b">
          <div className="flex items-center space-x-2">
            <MessageSquare className="h-4 w-4" />
            <h3 className="font-semibold">Chat with Data Processor</h3>
            <Badge variant="default">Active</Badge>
          </div>
        </div>
        
        <div className="flex-1 p-4 space-y-4 overflow-y-auto">
          <div className="flex items-start space-x-3">
            <Bot className="h-6 w-6 text-blue-600 mt-1" />
            <div className="flex-1">
              <div className="bg-accent rounded-lg p-3">
                <p className="text-sm">Hello! I'm ready to help you process financial data. What would you like me to analyze?</p>
              </div>
              <span className="text-xs text-muted-foreground">2 minutes ago</span>
            </div>
          </div>
          
          <div className="flex items-start space-x-3 justify-end">
            <div className="flex-1 text-right">
              <div className="bg-primary text-primary-foreground rounded-lg p-3 inline-block">
                <p className="text-sm">Can you analyze the Q3 expense reports?</p>
              </div>
              <div className="text-xs text-muted-foreground mt-1">1 minute ago</div>
            </div>
            <User className="h-6 w-6 text-muted-foreground mt-1" />
          </div>
          
          <div className="flex items-start space-x-3">
            <Bot className="h-6 w-6 text-blue-600 mt-1" />
            <div className="flex-1">
              <div className="bg-accent rounded-lg p-3">
                <p className="text-sm">I'll analyze your Q3 expense reports. Please upload the files or provide the data source location.</p>
              </div>
              <span className="text-xs text-muted-foreground">Just now</span>
            </div>
          </div>
        </div>
        
        <div className="p-4 border-t">
          <div className="flex space-x-2">
            <Input
              placeholder="Type your message..."
              className="flex-1"
            />
            <Button size="sm">
              <Send className="h-4 w-4" />
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}