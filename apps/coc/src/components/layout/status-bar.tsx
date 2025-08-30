import React from "react";
import { cn } from "../../lib/utils";
import { Button } from "../ui/button";
import { Separator } from "../ui/separator";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "../ui/tooltip";

interface StatusBarProps {
  className?: string;
  items?: StatusItem[];
  connectionStatus?: "connected" | "disconnected" | "connecting";
  systemStats?: {
    cpu: number;
    memory: number;
    disk: number;
  };
}

interface StatusItem {
  id: string;
  label: string;
  value?: string | number;
  status?: "success" | "warning" | "error" | "info";
  tooltip?: string;
  onClick?: () => void;
}

export function StatusBar({ className, items = [], connectionStatus = "connected", systemStats }: StatusBarProps) {
  const getConnectionStatusColor = () => {
    switch (connectionStatus) {
      case "connected": return "text-green-500";
      case "disconnected": return "text-red-500";
      case "connecting": return "text-yellow-500";
      default: return "text-muted-foreground";
    }
  };

  const getConnectionStatusText = () => {
    switch (connectionStatus) {
      case "connected": return "Connected";
      case "disconnected": return "Disconnected";
      case "connecting": return "Connecting...";
      default: return "Unknown";
    }
  };

  const getStatusColor = (status?: string) => {
    switch (status) {
      case "success": return "text-green-500";
      case "warning": return "text-yellow-500";
      case "error": return "text-red-500";
      case "info": return "text-blue-500";
      default: return "text-muted-foreground";
    }
  };

  return (
    <TooltipProvider>
      <div className={cn(
        "flex h-6 items-center justify-between border-t bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60 px-4 text-xs",
        className
      )}>
        <div className="flex items-center space-x-4">
          {/* Connection Status */}
          <Tooltip>
            <TooltipTrigger asChild>
              <div className="flex items-center space-x-1">
                <div className={cn("h-2 w-2 rounded-full", getConnectionStatusColor())} />
                <span className={getConnectionStatusColor()}>{getConnectionStatusText()}</span>
              </div>
            </TooltipTrigger>
            <TooltipContent>
              <p>Backend connection status</p>
            </TooltipContent>
          </Tooltip>

          {/* Custom Status Items */}
          {items.map((item, index) => (
            <React.Fragment key={item.id}>
              <Separator orientation="vertical" className="h-3" />
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={item.onClick}
                    className="h-auto p-0 text-xs hover:bg-transparent"
                  >
                    <span className={getStatusColor(item.status)}>
                      {item.label}: {item.value}
                    </span>
                  </Button>
                </TooltipTrigger>
                {item.tooltip && (
                  <TooltipContent>
                    <p>{item.tooltip}</p>
                  </TooltipContent>
                )}
              </Tooltip>
            </React.Fragment>
          ))}
        </div>

        <div className="flex items-center space-x-4">
          {/* System Stats */}
          {systemStats && (
            <>
              <Tooltip>
                <TooltipTrigger asChild>
                  <span className="text-muted-foreground">
                    CPU: {systemStats.cpu.toFixed(1)}%
                  </span>
                </TooltipTrigger>
                <TooltipContent>
                  <p>CPU Usage</p>
                </TooltipContent>
              </Tooltip>

              <Separator orientation="vertical" className="h-3" />

              <Tooltip>
                <TooltipTrigger asChild>
                  <span className="text-muted-foreground">
                    RAM: {systemStats.memory.toFixed(1)}%
                  </span>
                </TooltipTrigger>
                <TooltipContent>
                  <p>Memory Usage</p>
                </TooltipContent>
              </Tooltip>

              <Separator orientation="vertical" className="h-3" />

              <Tooltip>
                <TooltipTrigger asChild>
                  <span className="text-muted-foreground">
                    Disk: {systemStats.disk.toFixed(1)}%
                  </span>
                </TooltipTrigger>
                <TooltipContent>
                  <p>Disk Usage</p>
                </TooltipContent>
              </Tooltip>
            </>
          )}

          {/* Version Info */}
          <Separator orientation="vertical" className="h-3" />
          <span className="text-muted-foreground/60 font-mono">
            COC v1.0.0
          </span>
        </div>
      </div>
    </TooltipProvider>
  );
}