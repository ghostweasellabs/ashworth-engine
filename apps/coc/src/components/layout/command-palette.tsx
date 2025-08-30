import React, { useState, useEffect } from "react";
import { cn } from "../../lib/utils";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "../ui/dialog";
import { Input } from "../ui/input";
import { ScrollArea } from "../ui/scroll-area";
import { Button } from "../ui/button";
import { Separator } from "../ui/separator";
import { Search, Settings, Terminal } from "@carbon/icons-react";

interface CommandPaletteProps {
  isOpen: boolean;
  onClose: () => void;
  onExecute?: (command: Command) => void;
}

interface Command {
  id: string;
  label: string;
  description?: string;
  category: string;
  icon?: React.ReactNode;
  shortcut?: string;
  action: () => void;
}

const defaultCommands: Command[] = [
  {
    id: "goto-dashboard",
    label: "Go to Dashboard",
    description: "Navigate to the main dashboard",
    category: "Navigation",
    icon: <Terminal size={16} />,
    shortcut: "Ctrl+1",
    action: () => console.log("Navigate to dashboard"),
  },
  {
    id: "goto-files",
    label: "Go to File Manager",
    description: "Navigate to file management interface",
    category: "Navigation",
    icon: <Terminal size={16} />,
    shortcut: "Ctrl+2",
    action: () => console.log("Navigate to files"),
  },
  {
    id: "goto-processes",
    label: "Go to Process Monitor",
    description: "Navigate to process monitoring dashboard",
    category: "Navigation",
    icon: <Terminal size={16} />,
    shortcut: "Ctrl+3",
    action: () => console.log("Navigate to processes"),
  },
  {
    id: "toggle-theme",
    label: "Toggle Theme",
    description: "Switch between light and dark themes",
    category: "Settings",
    icon: <Settings size={16} />,
    shortcut: "Ctrl+Shift+T",
    action: () => console.log("Toggle theme"),
  },
  {
    id: "quick-search",
    label: "Quick Search",
    description: "Search across all files and content",
    category: "Search",
    icon: <Search size={16} />,
    shortcut: "Ctrl+Shift+F",
    action: () => console.log("Quick search"),
  },
  {
    id: "run-command",
    label: "Run Terminal Command",
    description: "Execute a command in the terminal",
    category: "Actions",
    icon: <Terminal size={16} />,
    shortcut: "Ctrl+Shift+P",
    action: () => console.log("Run command"),
  },
];

export function CommandPalette({ isOpen, onClose, onExecute }: CommandPaletteProps) {
  const [query, setQuery] = useState("");
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [filteredCommands, setFilteredCommands] = useState(defaultCommands);

  // Filter commands based on query
  useEffect(() => {
    if (!query.trim()) {
      setFilteredCommands(defaultCommands);
      setSelectedIndex(0);
      return;
    }

    const filtered = defaultCommands.filter(command =>
      command.label.toLowerCase().includes(query.toLowerCase()) ||
      command.description?.toLowerCase().includes(query.toLowerCase()) ||
      command.category.toLowerCase().includes(query.toLowerCase())
    );

    setFilteredCommands(filtered);
    setSelectedIndex(0);
  }, [query]);

  // Handle keyboard navigation
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (!isOpen) return;

      switch (e.key) {
        case "ArrowDown":
          e.preventDefault();
          setSelectedIndex(prev => 
            prev < filteredCommands.length - 1 ? prev + 1 : 0
          );
          break;
        case "ArrowUp":
          e.preventDefault();
          setSelectedIndex(prev => 
            prev > 0 ? prev - 1 : filteredCommands.length - 1
          );
          break;
        case "Enter":
          e.preventDefault();
          if (filteredCommands[selectedIndex]) {
            executeCommand(filteredCommands[selectedIndex]);
          }
          break;
        case "Escape":
          e.preventDefault();
          onClose();
          break;
      }
    };

    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [isOpen, filteredCommands, selectedIndex, onClose]);

  // Reset state when dialog opens/closes
  useEffect(() => {
    if (isOpen) {
      setQuery("");
      setSelectedIndex(0);
    }
  }, [isOpen]);

  const executeCommand = (command: Command) => {
    command.action();
    onExecute?.(command);
    onClose();
  };

  const groupedCommands = filteredCommands.reduce((acc, command) => {
    if (!acc[command.category]) {
      acc[command.category] = [];
    }
    acc[command.category].push(command);
    return acc;
  }, {} as Record<string, Command[]>);

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="max-w-2xl p-0">
        <DialogHeader className="px-6 pt-6 pb-0">
          <DialogTitle className="flex items-center gap-2">
            <Terminal size={20} />
            Command Palette
          </DialogTitle>
        </DialogHeader>

        <div className="px-6">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
            <Input
              placeholder="Type a command or search..."
              value={query}
              onChange={(e: React.ChangeEvent<HTMLInputElement>) => setQuery(e.target.value)}
              className="pl-10"
              autoFocus
            />
          </div>
        </div>

        <ScrollArea className="max-h-96 px-6 pb-6">
          {Object.entries(groupedCommands).map(([category, commands], categoryIndex) => (
            <div key={category} className={categoryIndex > 0 ? "mt-4" : ""}>
              <div className="mb-2 px-2 text-xs font-medium text-muted-foreground uppercase tracking-wider">
                {category}
              </div>
              <div className="space-y-1">
                {commands.map((command, commandIndex) => {
                  const globalIndex = Object.entries(groupedCommands)
                    .slice(0, categoryIndex)
                    .reduce((acc, [, cmds]) => acc + cmds.length, 0) + commandIndex;
                  
                  return (
                    <Button
                      key={command.id}
                      variant="ghost"
                      onClick={() => executeCommand(command)}
                      className={cn(
                        "w-full justify-start h-auto p-3 text-left",
                        selectedIndex === globalIndex && "bg-accent"
                      )}
                    >
                      <div className="flex items-center gap-3 w-full">
                        <div className="flex-shrink-0">
                          {command.icon}
                        </div>
                        <div className="flex-1 min-w-0">
                          <div className="font-medium">{command.label}</div>
                          {command.description && (
                            <div className="text-sm text-muted-foreground truncate">
                              {command.description}
                            </div>
                          )}
                        </div>
                        {command.shortcut && (
                          <div className="flex-shrink-0 text-xs text-muted-foreground font-mono bg-muted px-2 py-1 rounded">
                            {command.shortcut}
                          </div>
                        )}
                      </div>
                    </Button>
                  );
                })}
              </div>
              {categoryIndex < Object.keys(groupedCommands).length - 1 && (
                <Separator className="mt-3" />
              )}
            </div>
          ))}

          {filteredCommands.length === 0 && (
            <div className="py-8 text-center text-muted-foreground">
              <Search className="mx-auto h-8 w-8 mb-2 opacity-50" />
              <p>No commands found</p>
              <p className="text-sm">Try a different search term</p>
            </div>
          )}
        </ScrollArea>

        <div className="border-t px-6 py-3 text-xs text-muted-foreground">
          <div className="flex items-center justify-between">
            <span>Use ↑↓ to navigate, ↵ to select, ESC to close</span>
            <span className="font-mono">Ctrl+K</span>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}