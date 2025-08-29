"""Ashworth Engine FastAPI application."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.config.settings import settings
from src.api.routes import router as api_router
from src.api.health import perform_comprehensive_health_check
from src.api.models import HealthCheckResponse

app = FastAPI(
    title="Ashworth Engine",
    description="Multi-agent financial intelligence platform built on LangGraph",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {
            "name": "workflows",
            "description": "Workflow management and execution"
        },
        {
            "name": "health",
            "description": "System health and monitoring"
        }
    ]
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router, prefix="/api/v1", tags=["workflows"])


@app.get("/", tags=["health"])
async def root():
    """Root endpoint with system information."""
    return {
        "message": "Ashworth Engine API",
        "description": "Multi-agent financial intelligence platform",
        "version": "0.1.0",
        "status": "running",
        "environment": settings.environment,
        "llm_provider": settings.llm_provider,
        "ollama_host": settings.ollama_host,
        "docs_url": "/docs",
        "health_check_url": "/health"
    }


@app.get("/health", response_model=HealthCheckResponse, tags=["health"])
async def health_check():
    """Comprehensive health check endpoint."""
    try:
        health_data = await perform_comprehensive_health_check()
        
        # Return appropriate HTTP status based on health
        status_code = 200
        if health_data["status"] == "degraded":
            status_code = 207  # Multi-Status
        elif health_data["status"] == "unhealthy":
            status_code = 503  # Service Unavailable
        
        return JSONResponse(
            status_code=status_code,
            content={
                "status": health_data["status"],
                "timestamp": health_data["timestamp"].isoformat(),
                "services": {k: v["status"] for k, v in health_data["services"].items()},
                "version": "0.1.0"
            }
        )
        
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "error",
                "timestamp": "2024-01-01T00:00:00Z",
                "services": {"api": "error"},
                "version": "0.1.0",
                "error": str(e)
            }
        )


@app.get("/health/detailed", tags=["health"])
async def detailed_health_check():
    """Detailed health check with service-specific information."""
    try:
        health_data = await perform_comprehensive_health_check()
        return health_data
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Health check failed: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level=settings.log_level.lower()
    )