"""Health check utilities for external services."""

import asyncio
import httpx
from datetime import datetime
from typing import Dict, Any

from src.config.settings import settings


async def check_database_health() -> Dict[str, Any]:
    """Check database connectivity."""
    try:
        # TODO: Implement actual database health check when database is set up
        # For now, return a placeholder
        return {
            "status": "unknown",
            "message": "Database health check not implemented yet",
            "response_time_ms": 0
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "message": f"Database connection failed: {str(e)}",
            "response_time_ms": None
        }


async def check_ollama_health() -> Dict[str, Any]:
    """Check Ollama server connectivity."""
    try:
        start_time = datetime.utcnow()
        
        # Test connection to Ollama server
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{settings.ollama_host}/api/tags")
            
        end_time = datetime.utcnow()
        response_time = int((end_time - start_time).total_seconds() * 1000)
        
        if response.status_code == 200:
            data = response.json()
            models = [model.get("name", "unknown") for model in data.get("models", [])]
            return {
                "status": "healthy",
                "message": f"Connected to Ollama server with {len(models)} models available",
                "response_time_ms": response_time,
                "available_models": models[:5]  # Limit to first 5 models
            }
        else:
            return {
                "status": "unhealthy",
                "message": f"Ollama server returned status {response.status_code}",
                "response_time_ms": response_time
            }
            
    except httpx.TimeoutException:
        return {
            "status": "unhealthy",
            "message": "Ollama server connection timeout",
            "response_time_ms": None
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "message": f"Ollama connection failed: {str(e)}",
            "response_time_ms": None
        }


async def check_openai_health() -> Dict[str, Any]:
    """Check OpenAI API connectivity."""
    try:
        if not settings.openai_api_key:
            return {
                "status": "not_configured",
                "message": "OpenAI API key not configured",
                "response_time_ms": None
            }
        
        start_time = datetime.utcnow()
        
        # Test OpenAI API with a simple request
        async with httpx.AsyncClient(timeout=10.0) as client:
            headers = {
                "Authorization": f"Bearer {settings.openai_api_key}",
                "Content-Type": "application/json"
            }
            response = await client.get("https://api.openai.com/v1/models", headers=headers)
        
        end_time = datetime.utcnow()
        response_time = int((end_time - start_time).total_seconds() * 1000)
        
        if response.status_code == 200:
            return {
                "status": "healthy",
                "message": "OpenAI API accessible",
                "response_time_ms": response_time
            }
        else:
            return {
                "status": "unhealthy",
                "message": f"OpenAI API returned status {response.status_code}",
                "response_time_ms": response_time
            }
            
    except httpx.TimeoutException:
        return {
            "status": "unhealthy",
            "message": "OpenAI API connection timeout",
            "response_time_ms": None
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "message": f"OpenAI API connection failed: {str(e)}",
            "response_time_ms": None
        }


async def check_google_health() -> Dict[str, Any]:
    """Check Google AI API connectivity."""
    try:
        if not settings.google_api_key:
            return {
                "status": "not_configured",
                "message": "Google API key not configured",
                "response_time_ms": None
            }
        
        # For now, return a placeholder since Google AI API setup varies
        return {
            "status": "not_implemented",
            "message": "Google AI health check not implemented yet",
            "response_time_ms": None
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "message": f"Google AI connection failed: {str(e)}",
            "response_time_ms": None
        }


async def check_supabase_health() -> Dict[str, Any]:
    """Check Supabase connectivity."""
    try:
        if not settings.supabase_url:
            return {
                "status": "not_configured",
                "message": "Supabase URL not configured",
                "response_time_ms": None
            }
        
        start_time = datetime.utcnow()
        
        # Test Supabase REST API
        async with httpx.AsyncClient(timeout=5.0) as client:
            headers = {
                "apikey": settings.supabase_anon_key,
                "Authorization": f"Bearer {settings.supabase_anon_key}"
            }
            response = await client.get(f"{settings.supabase_url}/rest/v1/", headers=headers)
        
        end_time = datetime.utcnow()
        response_time = int((end_time - start_time).total_seconds() * 1000)
        
        if response.status_code in [200, 404]:  # 404 is OK for root endpoint
            return {
                "status": "healthy",
                "message": "Supabase API accessible",
                "response_time_ms": response_time
            }
        else:
            return {
                "status": "unhealthy",
                "message": f"Supabase API returned status {response.status_code}",
                "response_time_ms": response_time
            }
            
    except httpx.TimeoutException:
        return {
            "status": "unhealthy",
            "message": "Supabase connection timeout",
            "response_time_ms": None
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "message": f"Supabase connection failed: {str(e)}",
            "response_time_ms": None
        }


async def perform_comprehensive_health_check() -> Dict[str, Any]:
    """Perform health checks on all external services."""
    
    # Run all health checks concurrently
    health_checks = await asyncio.gather(
        check_database_health(),
        check_ollama_health(),
        check_openai_health(),
        check_google_health(),
        check_supabase_health(),
        return_exceptions=True
    )
    
    # Map results to service names
    services = {
        "database": health_checks[0] if not isinstance(health_checks[0], Exception) else {
            "status": "error", "message": str(health_checks[0]), "response_time_ms": None
        },
        "ollama": health_checks[1] if not isinstance(health_checks[1], Exception) else {
            "status": "error", "message": str(health_checks[1]), "response_time_ms": None
        },
        "openai": health_checks[2] if not isinstance(health_checks[2], Exception) else {
            "status": "error", "message": str(health_checks[2]), "response_time_ms": None
        },
        "google": health_checks[3] if not isinstance(health_checks[3], Exception) else {
            "status": "error", "message": str(health_checks[3]), "response_time_ms": None
        },
        "supabase": health_checks[4] if not isinstance(health_checks[4], Exception) else {
            "status": "error", "message": str(health_checks[4]), "response_time_ms": None
        }
    }
    
    # Determine overall health status
    healthy_services = sum(1 for service in services.values() 
                          if service.get("status") == "healthy")
    total_services = len(services)
    
    if healthy_services == total_services:
        overall_status = "healthy"
    elif healthy_services > 0:
        overall_status = "degraded"
    else:
        overall_status = "unhealthy"
    
    return {
        "status": overall_status,
        "timestamp": datetime.utcnow(),
        "services": services,
        "summary": {
            "healthy_services": healthy_services,
            "total_services": total_services,
            "health_percentage": round((healthy_services / total_services) * 100, 1)
        }
    }