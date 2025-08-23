#!/usr/bin/env python3
"""
Ashworth Engine v2 Server
Run the FastAPI application with uvicorn
"""

import uvicorn
from src.api.routes import app
from src.config.settings import settings

if __name__ == "__main__":
    print("🚀 Starting Ashworth Engine v2...")
    print(f"Environment: {settings.ae_env}")
    print(f"Supabase URL: {settings.supabase_url}")
    print("=" * 50)
    
    uvicorn.run(
        "src.api.routes:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )