import * as React from "react";
import {
  CommandDialog,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
  CommandSeparator,
  CommandShortcut,
} from "@/components/ui/command";
import {
  FileText,
  Activity,
  Package,
  GitBranch,
  MessageSquare,
  Database,
  Settings,
  Search,
  Terminal,
  Folder,
  Users,
  BarChart3,
  Shield,
  Zap,
} from "lucide-react";

export interface Command {
  id: string;
  label: string;
  description?: string;
  icon?: React.ComponentType<{ className?: string }>;
  shortcut?: string[];
  group: string;
  action: () => void;
}

interface CommandPaletteProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onNavigate?: (path: string) => void;
}

export function CommandPalette({ open, onOpenChange, onNavigate }: CommandPaletteProps) {
  const [search, setSearch] = React.useState("");

  // Define available commands
  const commands: Command[] = [
    // Navigation commands
    {
      id: "nav-dashboard",
      label: "Dashboard",
      description: "Go to main dashboard",
      icon: BarChart3,
      shortcut: ["d"],
      group: "Navigation",
      action: () => onNavigate?.("/dashboard"),
    },
    {
      id: "nav-files",
      label: "File Manager",
      description: "Browse and manage files",
      icon: Folder,
      shortcut: ["f"],
      group: "Navigation",
      action: () => onNavigate?.("/files"),
    },
    {
      id: "nav-processes",
      label: "Process Monitor",
      description: "Monitor system processes",
      icon: Activity,
      shortcut: ["p"],
      group: "Navigation",
      action: () => onNavigate?.("/processes"),
    },
    {
      id: "nav-packages",
      label: "Package Manager",
      description: "Manage packages and dependencies",
      icon: Package,
      shortcut: ["k"],
      group: "Navigation",
      action: () => onNavigate?.("/packages"),
    },
    {
      id: "nav-git",
      label: "Git Operations",
      description: "Git repository management",
      icon: GitBranch,
      shortcut: ["g"],
      group: "Navigation",
      action: () => onNavigate?.("/git"),
    },
    {
      id: "nav-agents",
      label: "Agent Chat",
      description: "Chat with AI agents",
      icon: MessageSquare,
      shortcut: ["a"],
      group: "Navigation",
      action: () => onNavigate?.("/agents"),
    },
    {
      id: "nav-rag",
      label: "RAG Management",
      description: "Manage knowledge bases",
      icon: Database,
      shortcut: ["r"],
      group: "Navigation",
      action: () => onNavigate?.("/rag"),
    },
    {
      id: "nav-settings",
      label: "Settings",
      description: "Application settings",
      icon: Settings,
      shortcut: ["s"],
      group: "Navigation",
      action: () => onNavigate?.("/settings"),
    },

    // Quick actions
    {
      id: "action-search",
      label: "Global Search",
      description: "Search across all content",
      icon: Search,
      shortcut: ["ctrl", "shift", "f"],
      group: "Actions",
      action: () => console.log("Global search"),
    },
    {
      id: "action-terminal",
      label: "Open Terminal",
      description: "Open integrated terminal",
      icon: Terminal,
      shortcut: ["ctrl", "`"],
      group: "Actions",
      action: () => console.log("Open terminal"),
    },
    {
      id: "action-new-file",
      label: "New File",
      description: "Create a new file",
      icon: FileText,
      shortcut: ["ctrl", "n"],
      group: "Actions",
      action: () => console.log("New file"),
    },

    // System commands
    {
      id: "system-reload",
      label: "Reload Application",
      description: "Reload the entire application",
      icon: Zap,
      shortcut: ["ctrl", "r"],
      group: "System",
      action: () => window.location.reload(),
    },
    {
      id: "system-security",
      label: "Security Center",
      description: "View security settings",
      icon: Shield,
      group: "System",
      action: () => onNavigate?.("/security"),
    },
  ];

  // Filter commands based on search
  const filteredCommands = React.useMemo(() => {
    if (!search) return commands;
    
    return commands.filter((command) =>
      command.label.toLowerCase().includes(search.toLowerCase()) ||
      command.description?.toLowerCase().includes(search.toLowerCase()) ||
      command.group.toLowerCase().includes(search.toLowerCase())
    );
  }, [search, commands]);

  // Group commands
  const groupedCommands = React.useMemo(() => {
    const groups: Record<string, Command[]> = {};
    
    filteredCommands.forEach((command) => {
      if (!groups[command.group]) {
        groups[command.group] = [];
      }
      groups[command.group].push(command);
    });
    
    return groups;
  }, [filteredCommands]);

  const handleSelect = React.useCallback((command: Command) => {
    command.action();
    onOpenChange(false);
    setSearch("");
  }, [onOpenChange]);

  React.useEffect(() => {
    if (!open) {
      setSearch("");
    }
  }, [open]);

  return (
    <CommandDialog open={open} onOpenChange={onOpenChange}>
      <CommandInput
        placeholder="Type a command or search..."
        value={search}
        onValueChange={setSearch}
      />
      <CommandList>
        <CommandEmpty>No results found.</CommandEmpty>
        
        {Object.entries(groupedCommands).map(([group, groupCommands], index) => (
          <React.Fragment key={group}>
            {index > 0 && <CommandSeparator />}
            <CommandGroup heading={group}>
              {groupCommands.map((command) => {
                const Icon = command.icon;
                return (
                  <CommandItem
                    key={command.id}
                    value={`${command.label} ${command.description || ""}`}
                    onSelect={() => handleSelect(command)}
                    className="flex items-center gap-2 px-2 py-1.5"
                  >
                    {Icon && <Icon className="h-4 w-4" />}
                    <div className="flex flex-col">
                      <span>{command.label}</span>
                      {command.description && (
                        <span className="text-xs text-muted-foreground">
                          {command.description}
                        </span>
                      )}
                    </div>
                    {command.shortcut && (
                      <CommandShortcut>
                        {command.shortcut.join("+")}
                      </CommandShortcut>
                    )}
                  </CommandItem>
                );
              })}
            </CommandGroup>
          </React.Fragment>
        ))}
      </CommandList>
    </CommandDialog>
  );
}