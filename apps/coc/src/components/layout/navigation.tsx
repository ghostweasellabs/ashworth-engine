import React from "react";
import { cn } from "../../lib/utils";
import { Button } from "../ui/button";
import { Separator } from "../ui/separator";
import { Tabs, TabsList, TabsTrigger } from "../ui/tabs";

interface NavigationProps {
  className?: string;
  items: NavigationItem[];
  activeItem?: string;
  onItemClick?: (itemId: string) => void;
  variant?: "tabs" | "buttons" | "breadcrumb";
}

interface NavigationItem {
  id: string;
  label: string;
  icon?: React.ReactNode;
  disabled?: boolean;
  badge?: number | string;
}

export function Navigation({ 
  className, 
  items, 
  activeItem, 
  onItemClick, 
  variant = "buttons" 
}: NavigationProps) {
  
  if (variant === "tabs") {
    return (
      <Tabs value={activeItem} onValueChange={onItemClick} className={className}>
        <TabsList className="grid w-full grid-cols-auto">
          {items.map((item) => (
            <TabsTrigger 
              key={item.id} 
              value={item.id}
              disabled={item.disabled}
              className="flex items-center gap-2"
            >
              {item.icon}
              <span>{item.label}</span>
              {item.badge && (
                <span className="ml-1 inline-flex items-center justify-center rounded-full bg-primary px-1.5 py-0.5 text-xs font-medium text-primary-foreground">
                  {item.badge}
                </span>
              )}
            </TabsTrigger>
          ))}
        </TabsList>
      </Tabs>
    );
  }

  if (variant === "breadcrumb") {
    return (
      <nav className={cn("flex items-center space-x-1 text-sm text-muted-foreground", className)}>
        {items.map((item, index) => (
          <React.Fragment key={item.id}>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => onItemClick?.(item.id)}
              disabled={item.disabled}
              className={cn(
                "h-auto p-1 font-normal",
                activeItem === item.id && "text-foreground font-medium"
              )}
            >
              {item.icon}
              <span className="ml-1">{item.label}</span>
            </Button>
            {index < items.length - 1 && (
              <span className="text-muted-foreground/50">/</span>
            )}
          </React.Fragment>
        ))}
      </nav>
    );
  }

  // Default buttons variant
  return (
    <nav className={cn("flex items-center space-x-2", className)}>
      {items.map((item, index) => (
        <React.Fragment key={item.id}>
          <Button
            variant={activeItem === item.id ? "default" : "ghost"}
            size="sm"
            onClick={() => onItemClick?.(item.id)}
            disabled={item.disabled}
            className="flex items-center gap-2"
          >
            {item.icon}
            <span>{item.label}</span>
            {item.badge && (
              <span className="ml-1 inline-flex items-center justify-center rounded-full bg-background/20 px-1.5 py-0.5 text-xs font-medium">
                {item.badge}
              </span>
            )}
          </Button>
          {index < items.length - 1 && variant === "buttons" && (
            <Separator orientation="vertical" className="h-4" />
          )}
        </React.Fragment>
      ))}
    </nav>
  );
}