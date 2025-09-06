"""LLM configuration management with environment-based model selection."""

import os
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

from .base import ModelTier

logger = logging.getLogger(__name__)


class ProviderType(Enum):
    """Available LLM providers."""
    OLLAMA = "ollama"
    OPENAI = "openai"
    GOOGLE = "google"


@dataclass
class ModelConfig:
    """Configuration for a specific model."""
    name: str
    provider: ProviderType
    tier: ModelTier
    cost_per_1k_tokens: float = 0.0
    max_tokens: int = 4096
    supports_streaming: bool = True
    description: str = ""


class LLMConfigManager:
    """Manages LLM configuration with environment-based overrides."""
    
    # Default model configurations
    DEFAULT_MODELS = {
        # OpenAI Models
        "gpt-5": ModelConfig(
            name="gpt-5",
            provider=ProviderType.OPENAI,
            tier=ModelTier.HEAVY,
            cost_per_1k_tokens=0.03,  # $30 per 1M tokens
            max_tokens=8192,
            description="GPT-5 thinking model with advanced reasoning"
        ),
        "gpt-4.1": ModelConfig(
            name="gpt-4.1",
            provider=ProviderType.OPENAI,
            tier=ModelTier.HEAVY,
            cost_per_1k_tokens=0.015,  # $15 per 1M tokens
            max_tokens=8192,
            description="GPT-4.1 with improved performance"
        ),
        "gpt-4.1-mini": ModelConfig(
            name="gpt-4.1-mini",
            provider=ProviderType.OPENAI,
            tier=ModelTier.LIGHT,
            cost_per_1k_tokens=0.001,  # $1 per 1M tokens
            max_tokens=4096,
            description="Efficient GPT-4.1 variant for light tasks"
        ),
        
        # Google Models
        "gemini-2.5-pro": ModelConfig(
            name="gemini-2.5-pro",
            provider=ProviderType.GOOGLE,
            tier=ModelTier.HEAVY,
            cost_per_1k_tokens=0.0125,  # Estimated
            max_tokens=8192,
            description="Google's most capable model"
        ),
        "gemini-2.5-flash": ModelConfig(
            name="gemini-2.5-flash",
            provider=ProviderType.GOOGLE,
            tier=ModelTier.LIGHT,
            cost_per_1k_tokens=0.0005,  # Estimated
            max_tokens=4096,
            description="Fast and efficient Gemini variant"
        ),
        
        # Ollama Models (Local)
        "gpt-oss:20b": ModelConfig(
            name="gpt-oss:20b",
            provider=ProviderType.OLLAMA,
            tier=ModelTier.HEAVY,
            cost_per_1k_tokens=0.0,  # Local models are free
            max_tokens=4096,
            description="Local GPT-style model for heavy reasoning"
        ),
        "gemma3n:e4b": ModelConfig(
            name="gemma3n:e4b",
            provider=ProviderType.OLLAMA,
            tier=ModelTier.LIGHT,
            cost_per_1k_tokens=0.0,
            max_tokens=2048,
            description="Efficient local model for light tasks"
        ),
        "llama3:70b": ModelConfig(
            name="llama3:70b",
            provider=ProviderType.OLLAMA,
            tier=ModelTier.HEAVY,
            cost_per_1k_tokens=0.0,
            max_tokens=4096,
            description="Large local model for complex reasoning"
        ),
        "phi3:mini": ModelConfig(
            name="phi3:mini",
            provider=ProviderType.OLLAMA,
            tier=ModelTier.LIGHT,
            cost_per_1k_tokens=0.0,
            max_tokens=2048,
            description="Small efficient local model"
        )
    }
    
    def __init__(self):
        """Initialize configuration manager."""
        self.models = self.DEFAULT_MODELS.copy()
        self._load_environment_config()
    
    def _load_environment_config(self) -> None:
        """Load configuration from environment variables."""
        # Override model preferences from environment
        heavy_pref = os.getenv("HEAVY_MODEL_PREFERENCE", "auto")
        light_pref = os.getenv("LIGHT_MODEL_PREFERENCE", "auto")
        
        if heavy_pref != "auto" and heavy_pref not in self.models:
            logger.warning(f"Heavy model preference '{heavy_pref}' not found in available models")
        
        if light_pref != "auto" and light_pref not in self.models:
            logger.warning(f"Light model preference '{light_pref}' not found in available models")
    
    def get_router_config(self) -> Dict[str, Any]:
        """Get complete router configuration."""
        return {
            # Provider configuration
            "ollama_host": os.getenv("OLLAMA_HOST", "http://192.168.1.220:11434"),
            "openai_api_key": os.getenv("OPENAI_API_KEY"),
            "google_api_key": os.getenv("GOOGLE_API_KEY"),
            
            # Model preferences
            "heavy_model_preference": os.getenv("HEAVY_MODEL_PREFERENCE", "auto"),
            "light_model_preference": os.getenv("LIGHT_MODEL_PREFERENCE", "auto"),
            
            # Fallback configuration
            "fallback_order": self._parse_fallback_order(),
            
            # GPT-5 specific configuration
            "gpt5_reasoning_effort": os.getenv("GPT5_REASONING_EFFORT", "medium"),
            "gpt5_include_reasoning": os.getenv("GPT5_INCLUDE_REASONING", "false").lower() == "true",
            "gpt5_reasoning_depth": os.getenv("GPT5_REASONING_DEPTH", "standard"),
            
            # Performance configuration
            "enable_metrics_logging": os.getenv("ENABLE_METRICS_LOGGING", "true").lower() == "true",
            "metrics_log_file": os.getenv("METRICS_LOG_FILE", "logs/llm_metrics.jsonl"),
            "performance_threshold_seconds": float(os.getenv("PERFORMANCE_THRESHOLD_SECONDS", "30.0")),
            "cost_threshold_dollars": float(os.getenv("COST_THRESHOLD_DOLLARS", "0.10")),
            
            # Health check configuration
            "health_check_interval_seconds": int(os.getenv("HEALTH_CHECK_INTERVAL_SECONDS", "300")),
            "provider_timeout_seconds": int(os.getenv("PROVIDER_TIMEOUT_SECONDS", "60")),
            "max_retries": int(os.getenv("MAX_RETRIES", "3")),
            "retry_delay_seconds": float(os.getenv("RETRY_DELAY_SECONDS", "1.0"))
        }
    
    def _parse_fallback_order(self) -> List[str]:
        """Parse fallback order from environment."""
        fallback_env = os.getenv("LLM_FALLBACK_ORDER", "ollama,openai,google")
        return [provider.strip() for provider in fallback_env.split(",")]
    
    def get_models_by_tier(self, tier: ModelTier) -> List[ModelConfig]:
        """Get all models for a specific tier."""
        return [model for model in self.models.values() if model.tier == tier]
    
    def get_models_by_provider(self, provider: ProviderType) -> List[ModelConfig]:
        """Get all models for a specific provider."""
        return [model for model in self.models.values() if model.provider == provider]
    
    def get_model_config(self, model_name: str) -> Optional[ModelConfig]:
        """Get configuration for a specific model."""
        return self.models.get(model_name)
    
    def get_recommended_model(
        self,
        tier: ModelTier,
        provider: Optional[ProviderType] = None,
        max_cost: Optional[float] = None
    ) -> Optional[ModelConfig]:
        """Get recommended model based on criteria."""
        candidates = self.get_models_by_tier(tier)
        
        # Filter by provider if specified
        if provider:
            candidates = [m for m in candidates if m.provider == provider]
        
        # Filter by cost if specified
        if max_cost is not None:
            candidates = [m for m in candidates if m.cost_per_1k_tokens <= max_cost]
        
        if not candidates:
            return None
        
        # Sort by cost (prefer lower cost) and return best option
        candidates.sort(key=lambda m: m.cost_per_1k_tokens)
        return candidates[0]
    
    def validate_configuration(self) -> Dict[str, Any]:
        """Validate current configuration and return status."""
        config = self.get_router_config()
        validation_results = {
            "valid": True,
            "warnings": [],
            "errors": [],
            "provider_status": {}
        }
        
        # Check provider configurations
        providers = {
            "ollama": {
                "required": ["ollama_host"],
                "optional": []
            },
            "openai": {
                "required": ["openai_api_key"],
                "optional": []
            },
            "google": {
                "required": ["google_api_key"],
                "optional": []
            }
        }
        
        for provider_name, requirements in providers.items():
            provider_valid = True
            provider_warnings = []
            
            # Check required configuration
            for req_key in requirements["required"]:
                if not config.get(req_key):
                    provider_valid = False
                    validation_results["errors"].append(
                        f"Missing required configuration for {provider_name}: {req_key}"
                    )
            
            # Check model preferences
            heavy_pref = config.get("heavy_model_preference")
            light_pref = config.get("light_model_preference")
            
            if heavy_pref != "auto" and heavy_pref not in self.models:
                provider_warnings.append(f"Heavy model preference '{heavy_pref}' not available")
            
            if light_pref != "auto" and light_pref not in self.models:
                provider_warnings.append(f"Light model preference '{light_pref}' not available")
            
            validation_results["provider_status"][provider_name] = {
                "valid": provider_valid,
                "warnings": provider_warnings
            }
            
            if not provider_valid:
                validation_results["valid"] = False
            
            validation_results["warnings"].extend(provider_warnings)
        
        # Check if at least one provider is configured
        if not any(status["valid"] for status in validation_results["provider_status"].values()):
            validation_results["valid"] = False
            validation_results["errors"].append("No valid providers configured")
        
        return validation_results
    
    def get_configuration_template(self) -> str:
        """Get environment variable template for configuration."""
        template = """
# LLM Provider Configuration
LLM_PROVIDER=ollama
OLLAMA_HOST=http://192.168.1.220:11434
OPENAI_API_KEY=your-openai-api-key-here
GOOGLE_API_KEY=your-google-api-key-here

# Model Selection Preferences
HEAVY_MODEL_PREFERENCE=auto  # auto, gpt-5, gpt-4.1, gemini-2.5-pro, gpt-oss:20b
LIGHT_MODEL_PREFERENCE=auto  # auto, gpt-4.1-mini, gemini-2.5-flash, gemma3n:e4b

# Provider Fallback Order
LLM_FALLBACK_ORDER=ollama,openai,google

# GPT-5 Specific Configuration
GPT5_REASONING_EFFORT=medium  # low, medium, high
GPT5_INCLUDE_REASONING=false
GPT5_REASONING_DEPTH=standard  # quick, standard, thorough

# Performance Monitoring
ENABLE_METRICS_LOGGING=true
METRICS_LOG_FILE=logs/llm_metrics.jsonl
PERFORMANCE_THRESHOLD_SECONDS=30.0
COST_THRESHOLD_DOLLARS=0.10

# Health Check Configuration
HEALTH_CHECK_INTERVAL_SECONDS=300
PROVIDER_TIMEOUT_SECONDS=60
MAX_RETRIES=3
RETRY_DELAY_SECONDS=1.0
"""
        return template.strip()


# Global configuration manager instance
_config_manager: Optional[LLMConfigManager] = None


def get_config_manager() -> LLMConfigManager:
    """Get global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = LLMConfigManager()
    return _config_manager


def get_router_config() -> Dict[str, Any]:
    """Convenience function to get router configuration."""
    return get_config_manager().get_router_config()