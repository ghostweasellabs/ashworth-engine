# Ashworth Engine Windows Setup Script
Write-Host "🚀 Setting up Ashworth Engine on Windows..." -ForegroundColor Green

# Check if uv is installed
if (!(Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "❌ uv is not installed. Please install uv first:" -ForegroundColor Red
    Write-Host "   Visit: https://docs.astral.sh/uv/getting-started/installation/" -ForegroundColor Yellow
    exit 1
}

# Install Python dependencies
Write-Host "📦 Installing Python dependencies..." -ForegroundColor Blue
uv sync

# Install pre-commit hooks
Write-Host "🔧 Setting up pre-commit hooks..." -ForegroundColor Blue
uv run pre-commit install

# Check if Supabase CLI is installed
if (!(Get-Command supabase -ErrorAction SilentlyContinue)) {
    Write-Host "🗄️ Installing Supabase CLI..." -ForegroundColor Blue
    
    # Install via Scoop if available, otherwise provide instructions
    if (Get-Command scoop -ErrorAction SilentlyContinue) {
        scoop bucket add supabase https://github.com/supabase/scoop-bucket.git
        scoop install supabase
    } else {
        Write-Host "❌ Supabase CLI not found. Please install it:" -ForegroundColor Red
        Write-Host "   Option 1: Install Scoop, then run: scoop bucket add supabase https://github.com/supabase/scoop-bucket.git && scoop install supabase" -ForegroundColor Yellow
        Write-Host "   Option 2: Download from: https://github.com/supabase/cli/releases" -ForegroundColor Yellow
        Write-Host "   Option 3: Use npm: npm install -g supabase" -ForegroundColor Yellow
        exit 1
    }
}

# Initialize Supabase (if not already initialized)
if (!(Test-Path "supabase/config.toml")) {
    Write-Host "🔧 Initializing Supabase..." -ForegroundColor Blue
    supabase init
}

# Start Supabase services
Write-Host "🚀 Starting Supabase services..." -ForegroundColor Blue
supabase start

# Test Ollama connection
Write-Host "🤖 Testing Ollama connection..." -ForegroundColor Blue
try {
    $response = Invoke-RestMethod -Uri "http://192.168.7.43:11434/api/tags" -Method Get -TimeoutSec 5
    Write-Host "✅ Ollama server is accessible" -ForegroundColor Green
} catch {
    Write-Host "⚠️ Warning: Ollama server not accessible at http://192.168.7.43:11434" -ForegroundColor Yellow
}

# Test Python environment
Write-Host "🐍 Testing Python environment..." -ForegroundColor Blue
try {
    uv run python -c "from src.config.settings import settings; print(f'✅ Settings loaded: {settings.environment}')"
} catch {
    Write-Host "⚠️ Warning: Could not test Python environment" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "🎉 Development environment setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "📋 Next steps:" -ForegroundColor Cyan
Write-Host "  - Supabase Studio: http://localhost:54321" -ForegroundColor White
Write-Host "  - PostgreSQL: localhost:54322" -ForegroundColor White
Write-Host "  - API will run on: http://localhost:8000" -ForegroundColor White
Write-Host ""
Write-Host "🚀 To start the API: uv run uvicorn main:app --reload --host 0.0.0.0 --port 8000" -ForegroundColor Green