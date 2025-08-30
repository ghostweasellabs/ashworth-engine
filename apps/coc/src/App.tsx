import React from "react";
import { ThemeProvider } from "./contexts/theme-context";
import { AppLayout } from "./components/layout";
import { Dashboard } from "./components/dashboard";

function App() {
  return (
    <ThemeProvider defaultTheme="system">
      <AppLayout>
        <Dashboard />
      </AppLayout>
    </ThemeProvider>
  );
}

export default App;