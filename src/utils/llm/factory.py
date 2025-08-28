"""Factory for creating LLM router instances."""

from typing import Dict, Any
from src.config.settings import settings
from .router import LLMRouter


def create_llm_router() -> LLMRouter:
    """Create LLM router instance with application settings."""
    config = {
        "ollama_host": settings.ollama_host,
        "openai_api_key": settings.openai_api_key,
        "google_api_key": settings.google_api_key,
        "fallback_order": settings.llm_fallback_order,
        "gpt5_reasoning_effort": settings.gpt5_reasoning_effort,
        "gpt5_include_reasoning": settings.gpt5_include_reasoning,
        "gpt5_reasoning_depth": settings.gpt5_reasoning_depth,
    }
    
    return LLMRouter(config)


# Global router instance
_router_instance = None


def get_llm_router() -> LLMRouter:
    """Get singleton LLM router instance."""
    global _router_instance
    if _router_instance is None:
        _router_instance = create_llm_router()
    return _router_instance