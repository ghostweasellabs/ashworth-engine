import React, { useState, useEffect } from "react";
import { cn } from "@/lib/utils";

interface ViewTransitionProps {
  children: React.ReactNode;
  transitionKey: string;
  className?: string;
}

export function ViewTransition({ children, transitionKey, className }: ViewTransitionProps) {
  const [isVisible, setIsVisible] = useState(true);
  const [currentKey, setCurrentKey] = useState(transitionKey);

  useEffect(() => {
    if (transitionKey !== currentKey) {
      // Start exit animation
      setIsVisible(false);
      
      // After exit animation completes, update content and start enter animation
      const timer = setTimeout(() => {
        setCurrentKey(transitionKey);
        setIsVisible(true);
      }, 150); // Half of the transition duration
      
      return () => clearTimeout(timer);
    }
  }, [transitionKey, currentKey]);

  return (
    <div
      className={cn(
        "transition-all duration-300 ease-in-out",
        isVisible 
          ? "opacity-100 translate-x-0" 
          : "opacity-0 translate-x-4",
        className
      )}
    >
      {children}
    </div>
  );
}