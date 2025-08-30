export type Theme = "light" | "dark" | "system";

export interface ThemeConfig {
  theme: Theme;
  actualTheme: "light" | "dark";
}

export interface ThemeContextType extends ThemeConfig {
  setTheme: (theme: Theme) => void;
}

export interface ThemeProviderProps {
  children: React.ReactNode;
  defaultTheme?: Theme;
}

// Local storage key for theme persistence
export const THEME_STORAGE_KEY = "coc-theme" as const;

// Valid theme values for validation
export const VALID_THEMES: readonly Theme[] = ["light", "dark", "system"] as const;