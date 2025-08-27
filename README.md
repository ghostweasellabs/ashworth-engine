# Ashworth Engine

Multi-agent financial intelligence platform built on LangGraph for automated financial document processing, analysis, and reporting.

## Features

- Multi-format document ingestion (CSV, Excel, PDF)
- LangGraph-orchestrated multi-agent workflow
- IRS compliance and tax categorization
- Professional financial reporting
- Local-first architecture for data security

## Quick Start

1. Install dependencies:
   ```bash
   uv sync
   ```

2. Copy environment configuration:
   ```bash
   cp .env.example .env
   ```

3. Run the application:
   ```bash
   uv run python main.py
   ```

## Project Structure

- `src/agents/` - LangGraph agent implementations
- `src/workflows/` - StateGraph workflow definitions
- `src/api/` - FastAPI routes and endpoints
- `src/config/` - Configuration management
- `src/models/` - Pydantic data models
- `src/utils/` - Utility functions
- `tests/` - Test suite

## Development

Install development dependencies:
```bash
uv sync --dev
```

Run tests:
```bash
uv run pytest
```

## License

Private - All rights reserved