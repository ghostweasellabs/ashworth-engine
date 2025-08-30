import React from "react";
import { useTheme } from "../../contexts/theme-context";
import { cn } from "../../lib/utils";
import { Button } from "../ui/button";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "../ui/tooltip";
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from "../ui/dropdown-menu";

interface HeaderProps {
  className?: string;
}

export function Header({ className }: HeaderProps) {
  const { theme, setTheme, actualTheme } = useTheme();

  const toggleTheme = () => {
    if (theme === "light") {
      setTheme("dark");
    } else if (theme === "dark") {
      setTheme("system");
    } else {
      setTheme("light");
    }
  };

  const getThemeIcon = () => {
    if (theme === "system") {
      return "◐";
    }
    return actualTheme === "dark" ? "◑" : "○";
  };

  const getThemeLabel = () => {
    if (theme === "system") {
      return `System (${actualTheme})`;
    }
    return theme === "dark" ? "Dark" : "Light";
  };

  return (
    <header className={cn(
      "glass backdrop-blur-xl border-0 shadow-lg",
      className
    )}>
      <div className="container flex h-16 items-center justify-between px-6">
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-3">
            <div className="h-10 w-10 rounded-xl bg-gradient-to-br from-primary to-primary/80 flex items-center justify-center shadow-lg">
              <span className="text-primary-foreground font-bold">COC</span>
            </div>
            <div>
              <h1 className="text-xl font-semibold bg-gradient-to-r from-foreground to-foreground/70 bg-clip-text text-transparent">
                Combat Operations Center
              </h1>
            </div>
          </div>
        </div>

        <div className="flex items-center space-x-4">
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={toggleTheme}
                  className="glass hover:scale-105 transition-all duration-200"
                >
                  <span className="text-base">{getThemeIcon()}</span>
                </Button>
              </TooltipTrigger>
              <TooltipContent>
                <p>Current theme: {getThemeLabel()}. Click to cycle through themes.</p>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        </div>
      </div>
    </header>
  );
}