import React from "react";
import { cn } from "../../lib/utils";

interface MainContentProps {
  children: React.ReactNode;
  className?: string;
}

export function MainContent({ children, className }: MainContentProps) {
  return (
    <main className={cn(
      "flex-1 overflow-auto",
      className
    )}>
      <div className="container mx-auto p-8">
        {children}
      </div>
    </main>
  );
}