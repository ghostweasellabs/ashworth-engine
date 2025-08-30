#!/bin/bash
set -e

echo "🚀 Setting up Ashworth Engine development environment..."

# Install Python dependencies
echo "📦 Installing Python dependencies..."
uv sync

# Install pre-commit hooks
echo "🔧 Setting up pre-commit hooks..."
uv run pre-commit install

# Install Supabase CLI for Linux container
echo "🗄️ Installing Supabase CLI..."
curl -fsSL https://supabase.com/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# Initialize Supabase (if not already initialized)
if [ ! -f "supabase/config.toml" ]; then
    echo "🔧 Initializing Supabase..."
    supabase init
fi

# Start Supabase services
echo "🚀 Starting Supabase services..."
supabase start

# Test Ollama connection
echo "🤖 Testing Ollama connection..."
if curl -s http://192.168.7.43:11434/api/tags > /dev/null; then
    echo "✅ Ollama server is accessible"
else
    echo "⚠️ Warning: Ollama server not accessible at http://192.168.7.43:11434"
fi

# Test Python environment
echo "🐍 Testing Python environment..."
uv run python -c "from src.config.settings import settings; print(f'✅ Settings loaded: {settings.environment}')"

echo "🎉 Development environment setup complete!"
echo ""
echo "📋 Next steps:"
echo "  - Supabase Studio: http://localhost:54321"
echo "  - PostgreSQL: localhost:54322"
echo "  - API will run on: http://localhost:8000"
echo ""
echo "🚀 To start the API: uv run uvicorn main:app --reload --host 0.0.0.0 --port 8000"