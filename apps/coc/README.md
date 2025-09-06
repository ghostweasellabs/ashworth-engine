# Combat Operations Center (COC)

A modern command center interface built with Deno, Vite, React, and IBM Carbon Design System.

## Features

- **Modern Stack**: Deno + Vite + React + TypeScript
- **Enterprise UI**: IBM Carbon Design System components
- **Theme Support**: Multiple Carbon themes (white, g10, g90, g100)
- **Responsive Design**: Mobile-first responsive layout
- **Hot Reload**: Fast development with Vite HMR

## Development

### Prerequisites

- [Deno](https://deno.land/) installed
- Node.js compatible environment for npm packages

### Getting Started

1. **Install dependencies**:
   ```bash
   deno install
   ```

2. **Start development server**:
   ```bash
   deno task dev
   ```

3. **Open browser**: Navigate to http://localhost:3001

### Available Scripts

- `deno task dev` - Start development server with hot reload
- `deno task build` - Build for production
- `deno task preview` - Preview production build
- `deno task lint` - Lint TypeScript/TSX files
- `deno task fmt` - Format code
- `deno task check` - Type check all files

## Project Structure

```
apps/coc/
├── src/
│   ├── App.tsx              # Main application component
│   └── styles/
│       └── index.scss       # Global styles with Carbon imports
├── deno.json                # Deno configuration and tasks
├── vite.config.ts           # Vite configuration
├── tsconfig.json            # TypeScript configuration
├── index.html               # HTML template
├── main.ts                  # Application entry point
├── dev.ts                   # Development server script
├── build.ts                 # Build script
└── preview.ts               # Preview server script
```

## Technology Stack

- **Runtime**: Deno 1.40+
- **Build Tool**: Vite 5.0+
- **Framework**: React 18.2+
- **Language**: TypeScript 5.3+
- **UI Library**: IBM Carbon Design System 1.59+
- **Styling**: SCSS with Carbon design tokens

## Carbon Design System

This project uses IBM Carbon Design System for consistent, enterprise-grade UI components:

- **Grid System**: CSS Grid-based responsive layout
- **Components**: Pre-built accessible components
- **Themes**: Support for light and dark themes
- **Design Tokens**: Consistent spacing, colors, and typography
- **Icons**: Carbon Icons React library

## Configuration

### Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
# Development server
VITE_PORT=3001
VITE_HOST=localhost

# API endpoints
VITE_API_BASE_URL=http://localhost:8000
VITE_WS_URL=ws://localhost:8000/ws

# Application settings
VITE_APP_NAME="Combat Operations Center"
VITE_DEFAULT_THEME=white
```

### Vite Configuration

The Vite configuration includes:
- React plugin for JSX support
- Path aliases (`@/` → `./src/`)
- Development server on port 3001
- Optimized build with code splitting
- SCSS preprocessing

### Deno Configuration

The `deno.json` includes:
- Import maps for npm packages
- TypeScript compiler options
- Formatting and linting rules
- Task definitions for development workflow

## Best Practices

### Carbon Design System Usage

- Use Carbon components instead of custom HTML elements
- Follow Carbon's Grid system for layouts
- Utilize Carbon design tokens for consistent styling
- Implement proper theme switching with Theme component

### Code Organization

- Keep components focused and single-purpose
- Use TypeScript for type safety
- Follow Carbon's accessibility guidelines
- Implement responsive design with Carbon's breakpoints

## Contributing

1. Follow the existing code style and formatting
2. Use Carbon components when available
3. Ensure responsive design across all breakpoints
4. Test theme switching functionality
5. Run linting and type checking before commits

## License

Part of the Ashworth Engine project.