import React, { createContext, useContext, useState, useCallback, ReactNode } from "react";

export interface Route {
  path: string;
  title: string;
  component: React.ComponentType;
}

interface RouterContextType {
  currentPath: string;
  navigate: (path: string) => void;
  routes: Route[];
  registerRoute: (route: Route) => void;
  getCurrentRoute: () => Route | undefined;
}

const RouterContext = createContext<RouterContextType | undefined>(undefined);

interface RouterProviderProps {
  children: ReactNode;
  initialPath?: string;
}

export function RouterProvider({ children, initialPath = "/dashboard" }: RouterProviderProps) {
  const [currentPath, setCurrentPath] = useState(initialPath);
  const [routes, setRoutes] = useState<Route[]>([]);

  const navigate = useCallback((path: string) => {
    setCurrentPath(path);
    // Update browser URL without page reload
    window.history.pushState({}, "", path);
  }, []);

  const registerRoute = useCallback((route: Route) => {
    setRoutes(prev => {
      const existing = prev.find(r => r.path === route.path);
      if (existing) {
        return prev.map(r => r.path === route.path ? route : r);
      }
      return [...prev, route];
    });
  }, []);

  const getCurrentRoute = useCallback(() => {
    return routes.find(route => route.path === currentPath);
  }, [routes, currentPath]);

  // Handle browser back/forward buttons
  React.useEffect(() => {
    const handlePopState = () => {
      setCurrentPath(window.location.pathname);
    };

    window.addEventListener("popstate", handlePopState);
    return () => window.removeEventListener("popstate", handlePopState);
  }, []);

  const value: RouterContextType = {
    currentPath,
    navigate,
    routes,
    registerRoute,
    getCurrentRoute,
  };

  return (
    <RouterContext.Provider value={value}>
      {children}
    </RouterContext.Provider>
  );
}

export function useRouter() {
  const context = useContext(RouterContext);
  if (context === undefined) {
    throw new Error("useRouter must be used within a RouterProvider");
  }
  return context;
}