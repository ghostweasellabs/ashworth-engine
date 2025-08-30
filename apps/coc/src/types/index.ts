/**
 * Core type definitions for the Combat Operations Center
 */

export interface AppState {
  theme: "light" | "dark";
  sidebarCollapsed: boolean;
  activeView: string;
}

export interface NavigationItem {
  id: string;
  label: string;
  icon: string;
  path: string;
  badge?: number;
  children?: NavigationItem[];
}

export interface Notification {
  id: string;
  type: "success" | "error" | "info" | "warning";
  title: string;
  message: string;
  timestamp: Date;
  dismissed?: boolean;
}