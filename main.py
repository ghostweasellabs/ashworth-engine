"""Ashworth Engine FastAPI application."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config.settings import settings

app = FastAPI(
    title="Ashworth Engine",
    description="Multi-agent financial intelligence platform",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Ashworth Engine API",
        "version": "0.1.0",
        "status": "running",
        "environment": settings.environment,
        "llm_provider": settings.llm_provider,
        "ollama_host": settings.ollama_host
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": "2024-01-01T00:00:00Z",
        "services": {
            "api": "healthy",
            "database": "unknown",  # TODO: Add database health check
            "ollama": "unknown"     # TODO: Add Ollama health check
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level=settings.log_level.lower()
    )