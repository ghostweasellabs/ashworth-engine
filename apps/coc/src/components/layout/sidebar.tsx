import React, { useState } from "react";
import { cn } from "../../lib/utils";
import { Button } from "../ui/button";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "../ui/tooltip";
import { ScrollArea } from "../ui/scroll-area";
import { Separator } from "../ui/separator";
import {
  Dashboard,
  Folder,
  Activity,
  Package,
  Branch,
  Chat,
  Search,
  ChevronLeft,
  ChevronRight,
} from "@carbon/icons-react";

interface SidebarProps {
  className?: string;
}

interface NavigationItem {
  id: string;
  label: string;
  icon: string;
  path: string;
  badge?: number;
  children?: NavigationItem[];
}

const navigationItems: NavigationItem[] = [
  {
    id: "dashboard",
    label: "Dashboard",
    icon: "Dashboard",
    path: "/dashboard",
  },
  {
    id: "files",
    label: "File Manager",
    icon: "Folder",
    path: "/files",
  },
  {
    id: "processes",
    label: "Process Monitor",
    icon: "Activity",
    path: "/processes",
  },
  {
    id: "packages",
    label: "Package Manager",
    icon: "Package",
    path: "/packages",
  },
  {
    id: "git",
    label: "Git Operations",
    icon: "Branch",
    path: "/git",
  },
  {
    id: "agents",
    label: "Agent Chat",
    icon: "Chat",
    path: "/agents",
  },
  {
    id: "rag",
    label: "RAG Management",
    icon: "Search",
    path: "/rag",
  },
];

export function Sidebar({ className }: SidebarProps) {
  const [collapsed, setCollapsed] = useState(false);
  const [activeItem, setActiveItem] = useState("dashboard");

  const renderIcon = (iconName: string) => {
    const iconProps = { size: 20 };
    switch (iconName) {
      case "Dashboard": return <Dashboard {...iconProps} />;
      case "Folder": return <Folder {...iconProps} />;
      case "Activity": return <Activity {...iconProps} />;
      case "Package": return <Package {...iconProps} />;
      case "Branch": return <Branch {...iconProps} />;
      case "Chat": return <Chat {...iconProps} />;
      case "Search": return <Search {...iconProps} />;
      default: return null;
    }
  };

  return (
    <TooltipProvider>
      <aside className={cn(
        "glass backdrop-blur-xl transition-all duration-300 border-0 shadow-xl",
        collapsed ? "w-16" : "w-64",
        className
      )}>
        <div className="flex h-full flex-col">
          {/* Sidebar Header */}
          <div className="flex h-16 items-center justify-between px-4">
            {!collapsed && (
              <span className="text-sm font-medium text-muted-foreground/80">
                Navigation
              </span>
            )}
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={() => setCollapsed(!collapsed)}
                  className="glass hover:scale-105 transition-all duration-200 h-8 w-8"
                >
                  {collapsed ? <ChevronRight size={16} /> : <ChevronLeft size={16} />}
                </Button>
              </TooltipTrigger>
              <TooltipContent side="right">
                <p>{collapsed ? "Expand sidebar" : "Collapse sidebar"}</p>
              </TooltipContent>
            </Tooltip>
          </div>

          <Separator className="mx-4" />

          {/* Navigation Items */}
          <ScrollArea className="flex-1 px-3">
            <nav className="space-y-2 py-4">
              {navigationItems.map((item) => (
                <Tooltip key={item.id} delayDuration={collapsed ? 0 : 1000}>
                  <TooltipTrigger asChild>
                    <Button
                      variant={activeItem === item.id ? "secondary" : "ghost"}
                      onClick={() => setActiveItem(item.id)}
                      className={cn(
                        "w-full justify-start transition-all duration-200 hover:scale-[1.02]",
                        activeItem === item.id && "bg-gradient-to-r from-primary/20 to-primary/10 text-primary shadow-lg glass",
                        collapsed ? "px-2" : "px-3"
                      )}
                    >
                      <div className="w-5 h-5 flex items-center justify-center">
                        {renderIcon(item.icon)}
                      </div>
                      {!collapsed && (
                        <>
                          <span className="ml-3">{item.label}</span>
                          {item.badge && (
                            <span className="ml-auto inline-flex items-center justify-center rounded-full bg-primary px-2 py-1 text-xs font-medium text-primary-foreground">
                              {item.badge}
                            </span>
                          )}
                        </>
                      )}
                    </Button>
                  </TooltipTrigger>
                  {collapsed && (
                    <TooltipContent side="right">
                      <p>{item.label}</p>
                    </TooltipContent>
                  )}
                </Tooltip>
              ))}
            </nav>
          </ScrollArea>

          <Separator className="mx-4" />

          {/* Sidebar Footer */}
          <div className="p-4">
            {!collapsed ? (
              <div className="text-xs text-muted-foreground/60 space-y-1">
                <div>COC v1.0.0</div>
                <div className="font-mono">Deno + React</div>
              </div>
            ) : (
              <div className="flex justify-center">
                <span className="text-xs text-muted-foreground/60">v1.0</span>
              </div>
            )}
          </div>
        </div>
      </aside>
    </TooltipProvider>
  );
}