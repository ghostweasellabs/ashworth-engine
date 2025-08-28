# Simple Supabase Setup for Ashworth Engine
Write-Host "🗄️ Setting up Supabase..." -ForegroundColor Green

# Check if Supabase CLI is installed
if (!(Get-Command supabase -ErrorAction SilentlyContinue)) {
    Write-Host "❌ Supabase CLI not found. Install it first:" -ForegroundColor Red
    Write-Host "   npm install -g supabase" -ForegroundColor Yellow
    Write-Host "   OR download from: https://github.com/supabase/cli/releases" -ForegroundColor Yellow
    exit 1
}

Write-Host "✅ Supabase CLI found" -ForegroundColor Green

# Initialize if needed
if (!(Test-Path "supabase/config.toml")) {
    Write-Host "🔧 Initializing Supabase..." -ForegroundColor Blue
    supabase init
}

# Start Supabase
Write-Host "🚀 Starting Supabase..." -ForegroundColor Blue
supabase start

Write-Host ""
Write-Host "🎉 Supabase is running!" -ForegroundColor Green
Write-Host "  - Studio: http://localhost:54323" -ForegroundColor White
Write-Host "  - API: http://localhost:54321" -ForegroundColor White  
Write-Host "  - DB: localhost:54322" -ForegroundColor White
Write-Host ""
Write-Host "To stop: supabase stop" -ForegroundColor Yellow