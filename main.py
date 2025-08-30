"""Ashworth Engine FastAPI application."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.config.settings import settings
from src.api.routes import router as api_router
from src.api.health import perform_comprehensive_health_check
from src.api.models import HealthCheckResponse

app = FastAPI(
    title="Ashworth Engine API",
    description="""
    ## Multi-Agent Financial Intelligence Platform
    
    The Ashworth Engine is a sophisticated financial analysis platform that processes 
    financial documents through specialized AI agents to produce executive-grade reports 
    and insights.
    
    ### Key Features
    
    * **Multi-Agent Processing**: Specialized agents for data extraction, processing, 
      categorization, and report generation
    * **Real-time Progress Tracking**: Monitor workflow progress with agent-level status
    * **Flexible File Support**: Process CSV, Excel, and PDF financial documents
    * **IRS Compliance**: Tax categorization following current IRS guidelines
    * **Executive Reports**: Professional-grade financial analysis and recommendations
    * **Workflow Management**: Full lifecycle management with cancellation and interrupts
    
    ### Workflow Process
    
    1. **Data Fetcher Agent** (Dr. Marcus Thornfield) - Extracts data from uploaded files
    2. **Data Processor Agent** (Dexter Blackwood) - Cleans and validates financial data  
    3. **Categorizer Agent** (Clarke Pemberton) - Applies tax categorization and compliance
    4. **Report Generator Agent** (Prof. Elena Castellanos) - Creates executive reports
    
    ### Getting Started
    
    1. Upload financial documents using `POST /api/v1/workflows`
    2. Monitor progress with `GET /api/v1/workflows/{workflow_id}`
    3. Retrieve results with `GET /api/v1/workflows/{workflow_id}/results`
    4. Check system health with `GET /health`
    
    ### File Upload Requirements
    
    * **Supported Formats**: CSV (.csv), Excel (.xlsx, .xls), PDF (.pdf)
    * **Size Limits**: 50MB per file, 200MB total per workflow
    * **Content**: Financial transactions, receipts, statements, expense reports
    
    ### Error Handling
    
    The API uses standard HTTP status codes and provides detailed error messages:
    
    * `200` - Success
    * `202` - Accepted (workflow in progress)
    * `400` - Bad Request (validation errors)
    * `404` - Not Found
    * `413` - File too large
    * `415` - Unsupported file type
    * `422` - Validation failed
    * `500` - Internal server error
    * `503` - Service unavailable
    
    ### Rate Limits
    
    * Workflow creation: 10 requests per minute per client
    * Status checks: 100 requests per minute per client
    * File uploads: 100MB per minute per client
    """,
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    contact={
        "name": "Ashworth Engine Support",
        "email": "support@ashworth-engine.com",
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
    },
    openapi_tags=[
        {
            "name": "workflows",
            "description": """
            **Workflow Management and Execution**
            
            Create, monitor, and manage financial analysis workflows. Each workflow 
            processes uploaded financial documents through multiple specialized agents 
            to produce comprehensive financial insights and reports.
            
            Key endpoints:
            * Create workflows with file uploads
            * Monitor real-time progress and agent status  
            * Retrieve complete or partial results
            * Cancel or interrupt workflows for human review
            * List and filter workflows with pagination
            """
        },
        {
            "name": "health",
            "description": """
            **System Health and Monitoring**
            
            Monitor the health and connectivity of all system components including 
            databases, LLM providers, and external services. Essential for production 
            monitoring and troubleshooting.
            
            Health checks include:
            * Database connectivity (PostgreSQL/Supabase)
            * LLM provider status (Ollama, OpenAI, Google)
            * Service response times and availability
            * Overall system health scoring
            """
        }
    ],
    servers=[
        {
            "url": "http://localhost:8000",
            "description": "Development server"
        },
        {
            "url": "https://api.ashworth-engine.com",
            "description": "Production server"
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