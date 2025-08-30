import React from "react";
import { Header } from "./header";
import { Sidebar } from "./sidebar";
import { MainContent } from "./main-content";
import { cn } from "../../lib/utils";

interface AppLayoutProps {
  children: React.ReactNode;
  className?: string;
}

export function AppLayout({ children, className }: AppLayoutProps) {
  return (
    <div className={cn(
      "min-h-screen text-foreground flex flex-col",
      className
    )}>
      <Header />
      <div className="flex flex-1 overflow-hidden">
        <Sidebar />
        <MainContent>
          {children}
        </MainContent>
      </div>
    </div>
  );
}