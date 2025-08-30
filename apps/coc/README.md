# Combat Operations Center (COC)

A modern, visually stunning web-based command center that provides all the functionality of command line operations through an intuitive, powerful interface.

## Technology Stack

- **Runtime**: Deno 2.0+ with native TypeScript support
- **Build Tool**: Vite for lightning-fast development and optimized builds
- **UI Framework**: React 18+ with shadcn/ui components
- **Styling**: Tailwind CSS with custom design system
- **State Management**: Zustand (to be implemented)

## Getting Started

### Prerequisites

- Deno 2.0 or later installed
- Access to the existing Ashworth Engine monorepo

### Development

```bash
# Start development server
deno task dev

# Build for production
deno task build

# Preview production build
deno task preview

# Code quality
deno task lint
deno task fmt
deno task type-check
```

### Project Structure

```
apps/coc/
├── src/
│   ├── components/     # React components
│   ├── lib/           # Utility functions
│   ├── hooks/         # Custom React hooks
│   ├── types/         # TypeScript type definitions
│   ├── stores/        # State management
│   ├── App.tsx        # Main application component
│   ├── main.tsx       # React entry point
│   └── globals.css    # Global styles
├── public/            # Static assets
├── deno.json          # Deno configuration
├── vite.config.ts     # Vite configuration
├── tailwind.config.js # Tailwind configuration
└── index.html         # HTML entry point
```

## Features (Planned)

- 🗂️ File Management Interface
- 📊 Process Monitoring Dashboard  
- 📦 Package Management GUI
- 🔄 Git Operations Interface
- 🤖 Agent Chat System
- 🧠 RAG Management Tools
- 🌓 Light/Dark Theme Support
- ⌨️ Keyboard Shortcuts & Command Palette
- 🔄 Real-time Updates via WebSocket

## Integration

This application integrates with the existing Ashworth Engine infrastructure:

- **Backend**: FastAPI endpoints for all operations
- **Database**: Supabase for data persistence
- **Authentication**: Existing auth system
- **Agents**: LangGraph multi-agent workflows

## Development Guidelines

- Follow the established monorepo patterns
- Use TypeScript for all code
- Implement responsive design with Tailwind
- Maintain professional, enterprise-grade UI
- No emojis in the production interface
- Focus on exceptional user experience