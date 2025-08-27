# Supabase Setup Script for Windows
Write-Host "🗄️ Setting up Supabase for Ashworth Engine..." -ForegroundColor Green

# Check if Supabase CLI is installed
if (!(Get-Command supabase -ErrorAction SilentlyContinue)) {
    Write-Host "📦 Installing Supabase CLI..." -ForegroundColor Blue
    
    # Try different installation methods
    if (Get-Command scoop -ErrorAction SilentlyContinue) {
        Write-Host "Using Scoop to install Supabase CLI..." -ForegroundColor Yellow
        scoop bucket add supabase https://github.com/supabase/scoop-bucket.git
        scoop install supabase
    } elseif (Get-Command winget -ErrorAction SilentlyContinue) {
        Write-Host "Using winget to install Supabase CLI..." -ForegroundColor Yellow
        winget install --id Supabase.cli
    } else {
        Write-Host "❌ Neither Scoop nor winget found. Please install Supabase CLI manually:" -ForegroundColor Red
        Write-Host "   1. Download from: https://github.com/supabase/cli/releases" -ForegroundColor Yellow
        Write-Host "   2. Or install via npm: npm install -g supabase" -ForegroundColor Yellow
        Write-Host "   3. Or install Scoop first: https://scoop.sh" -ForegroundColor Yellow
        exit 1
    }
}

# Verify installation
if (!(Get-Command supabase -ErrorAction SilentlyContinue)) {
    Write-Host "❌ Supabase CLI installation failed. Please install manually." -ForegroundColor Red
    exit 1
}

Write-Host "✅ Supabase CLI is installed" -ForegroundColor Green

# Check if already initialized
if (Test-Path "supabase/config.toml") {
    Write-Host "✅ Supabase already initialized" -ForegroundColor Green
} else {
    Write-Host "🔧 Initializing Supabase..." -ForegroundColor Blue
    supabase init
}

# Start Supabase services
Write-Host "🚀 Starting Supabase services..." -ForegroundColor Blue
supabase start

# Check if services are running
Write-Host "🔍 Checking Supabase services..." -ForegroundColor Blue
try {
    $response = Invoke-RestMethod -Uri "http://localhost:54321/rest/v1/" -Method Get -TimeoutSec 10
    Write-Host "✅ Supabase API is running" -ForegroundColor Green
} catch {
    Write-Host "⚠️ Warning: Could not connect to Supabase API" -ForegroundColor Yellow
}

# Test database connection
Write-Host "🔍 Testing database connection..." -ForegroundColor Blue
try {
    $env:PGPASSWORD = "postgres"
    $result = & psql -h localhost -p 54322 -U postgres -d postgres -c "SELECT version();" 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ PostgreSQL is running" -ForegroundColor Green
    } else {
        Write-Host "⚠️ Warning: Could not connect to PostgreSQL" -ForegroundColor Yellow
    }
} catch {
    Write-Host "⚠️ Warning: psql not found, cannot test database connection" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "🎉 Supabase setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "📋 Services running:" -ForegroundColor Cyan
Write-Host "  - Supabase Studio: http://localhost:54323" -ForegroundColor White
Write-Host "  - API: http://localhost:54321" -ForegroundColor White
Write-Host "  - PostgreSQL: localhost:54322" -ForegroundColor White
Write-Host "  - Inbucket (Email): http://localhost:54324" -ForegroundColor White
Write-Host ""
Write-Host "🔧 To stop services: supabase stop" -ForegroundColor Yellow
Write-Host "🔧 To reset database: supabase db reset" -ForegroundColor Yellow