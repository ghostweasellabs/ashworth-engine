# Ashworth Engine v2

Multi-agent financial intelligence platform built on LangGraph with external Ollama integration.

## Quick Start

### Prerequisites
- Python 3.11+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) for Python package management
- [Supabase CLI](https://supabase.com/docs/guides/cli) for local database
- External Ollama server running at `192.168.7.43:11434`

### Setup

1. **Install dependencies**
   ```bash
   uv sync
   ```

2. **Start Supabase**
   ```bash
   # Windows
   .\setup-supabase.ps1
   
   # Or manually
   supabase start
   ```

3. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env if needed
   ```

4. **Run the API**
   ```bash
   uv run uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

## Services

When running, you'll have:
- **API**: http://localhost:8000
- **Supabase Studio**: http://localhost:54323
- **PostgreSQL**: localhost:54322
- **External Ollama**: http://192.168.7.43:11434

## Architecture

- **Multi-Agent System**: LangGraph orchestrates specialized financial agents
- **External LLM**: Uses external Ollama server (gpt-oss:20b model)
- **Database**: Supabase PostgreSQL with pgvector for RAG
- **State Management**: Persistent checkpoints and agent memory
- **File Processing**: Handles Excel, CSV, PDF uploads

## Development

Use the dev container for a complete development environment:
```bash
# Open in VS Code dev container
code .
```

Or run locally:
```bash
# Start Supabase
supabase start

# Run API
uv run uvicorn main:app --reload

# Run tests
uv run python -m pytest
```

## Agents

- **Dr. Marcus Thornfield** (Data Fetcher): Economist-driven data extraction
- **Dexter Blackwood** (Data Processor): Fraud detection and data cleaning  
- **Clarke Pemberton** (Categorizer): IRS compliance and tax optimization
- **Professor Elena Castellanos** (Report Generator): Executive storytelling
- **Dr. Victoria Ashworth** (Orchestrator): Workflow coordination

## Tech Stack

- **Backend**: FastAPI + LangGraph + PostgreSQL
- **LLM**: External Ollama (gpt-oss:20b)
- **Database**: Supabase (PostgreSQL + pgvector)
- **Package Management**: uv
- **Containerization**: Docker + Dev Containers