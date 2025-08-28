"""Factory for creating LLM router instances."""

from typing import Dict, Any
from src.config.settings import settings
from .router import LLMRouter
from .config_manager import get_router_config


def create_llm_router() -> LLMRouter:
    """Create LLM router instance with comprehensive configuration."""
    # Get configuration from the config manager (includes environment overrides)
    config = get_router_config()
    
    return LLMRouter(config)


# Global router instance
_router_instance = None


def get_llm_router() -> LLMRouter:
    """Get singleton LLM router instance."""
    global _router_instance
    if _router_instance is None:
        _router_instance = create_llm_router()
    return _router_instance