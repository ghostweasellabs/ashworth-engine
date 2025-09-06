import React from 'react';
import { RouterProvider } from 'react-router-dom';
import { ThemeProvider } from '@/contexts/ThemeContext.tsx';
import { LayoutProvider } from '@/contexts/LayoutContext.tsx';
import { router } from '@/router/index.tsx';

function App() {
  return (
    <ThemeProvider>
      <LayoutProvider>
        <RouterProvider router={router} />
      </LayoutProvider>
    </ThemeProvider>
  );
}

export default App;