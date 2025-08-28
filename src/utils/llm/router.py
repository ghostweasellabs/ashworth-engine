"""LLM router for intelligent provider selection and fallback handling."""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass

from .base import BaseLLMProvider, LLMResponse, ModelTier
from .providers import OpenAIProvider, GoogleProvider, OllamaProvider

logger = logging.getLogger(__name__)


@dataclass
class ProviderConfig:
    """Configuration for a specific provider."""
    provider_class: type
    config: Dict[str, Any]
    priority: int = 1  # Lower numbers = higher priority
    enabled: bool = True


class LLMRouter:
    """Intelligent LLM router with tier-based selection and fallback."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize router with provider configurations."""
        self.config = config
        self.providers: Dict[str, BaseLLMProvider] = {}
        self.provider_health: Dict[str, bool] = {}
        self.fallback_order = config.get("fallback_order", ["ollama", "openai", "google"])
        
        # Initialize providers based on configuration
        self._initialize_providers()
    
    def _initialize_providers(self) -> None:
        """Initialize all configured providers."""
        provider_configs = {
            "ollama": ProviderConfig(
                provider_class=OllamaProvider,
                config={"host": self.config.get("ollama_host", "http://192.168.7.43:11434")},
                priority=1,  # Highest priority for local-first approach
                enabled=True
            ),
            "openai": ProviderConfig(
                provider_class=OpenAIProvider,
                config={"api_key": self.config.get("openai_api_key")},
                priority=2,
                enabled=bool(self.config.get("openai_api_key"))
            ),
            "google": ProviderConfig(
                provider_class=GoogleProvider,
                config={"api_key": self.config.get("google_api_key")},
                priority=3,
                enabled=bool(self.config.get("google_api_key"))
            )
        }
        
        for name, provider_config in provider_configs.items():
            if provider_config.enabled:
                try:
                    self.providers[name] = provider_config.provider_class(provider_config.config)
                    logger.info(f"Initialized {name} provider")
                except Exception as e:
                    logger.warning(f"Failed to initialize {name} provider: {e}")
    
    async def generate(
        self,
        prompt: str,
        tier: ModelTier = ModelTier.LIGHT,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate response with intelligent provider selection."""
        
        # If specific provider is requested, try it first
        if provider and provider in self.providers:
            try:
                return await self._generate_with_provider(
                    provider, prompt, tier, model, temperature, max_tokens, **kwargs
                )
            except Exception as e:
                logger.warning(f"Requested provider {provider} failed: {e}")
                # Continue to fallback logic
        
        # Try providers in fallback order
        for provider_name in self.fallback_order:
            if provider_name not in self.providers:
                continue
                
            try:
                return await self._generate_with_provider(
                    provider_name, prompt, tier, model, temperature, max_tokens, **kwargs
                )
            except Exception as e:
                logger.warning(f"Provider {provider_name} failed: {e}")
                continue
        
        raise RuntimeError("All LLM providers failed")
    
    async def _generate_with_provider(
        self,
        provider_name: str,
        prompt: str,
        tier: ModelTier,
        model: Optional[str],
        temperature: float,
        max_tokens: Optional[int],
        **kwargs
    ) -> LLMResponse:
        """Generate response with a specific provider."""
        provider = self.providers[provider_name]
        
        # Select model if not specified
        if not model:
            model = provider.get_model_for_tier(tier)
        
        # Handle GPT-5 specific configuration
        if model.startswith("gpt-5"):
            kwargs.setdefault("reasoning_effort", self.config.get("gpt5_reasoning_effort", "medium"))
            kwargs.setdefault("include_reasoning", self.config.get("gpt5_include_reasoning", False))
            kwargs.setdefault("reasoning_depth", self.config.get("gpt5_reasoning_depth", "standard"))
        
        return await provider.generate(prompt, model, temperature, max_tokens, **kwargs)
    
    async def health_check_all(self) -> Dict[str, bool]:
        """Check health of all providers."""
        health_results = {}
        
        for name, provider in self.providers.items():
            try:
                health_results[name] = await provider.health_check()
            except Exception as e:
                logger.error(f"Health check failed for {name}: {e}")
                health_results[name] = False
        
        self.provider_health = health_results
        return health_results
    
    def get_available_models(self) -> Dict[str, Dict[ModelTier, List[str]]]:
        """Get all available models from all providers."""
        all_models = {}
        
        for name, provider in self.providers.items():
            try:
                all_models[name] = provider.get_available_models()
            except Exception as e:
                logger.warning(f"Failed to get models from {name}: {e}")
                all_models[name] = {ModelTier.HEAVY: [], ModelTier.LIGHT: []}
        
        return all_models
    
    def get_provider_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status information for all providers."""
        status = {}
        
        for name, provider in self.providers.items():
            status[name] = {
                "available": name in self.provider_health and self.provider_health[name],
                "models": provider.get_available_models(),
                "config": {
                    "host" if name == "ollama" else "api_key_configured": 
                    bool(provider.config.get("host" if name == "ollama" else "api_key"))
                }
            }
        
        return status
    
    async def benchmark_providers(
        self,
        test_prompt: str = "What is 2+2? Respond with just the number.",
        tier: ModelTier = ModelTier.LIGHT
    ) -> Dict[str, Dict[str, Any]]:
        """Benchmark all providers with a simple test prompt."""
        results = {}
        
        for name in self.providers.keys():
            try:
                response = await self._generate_with_provider(
                    name, test_prompt, tier, None, 0.1, 10
                )
                results[name] = {
                    "success": True,
                    "response_time": response.response_time,
                    "model": response.model,
                    "cost": response.cost,
                    "tokens": response.tokens_used
                }
            except Exception as e:
                results[name] = {
                    "success": False,
                    "error": str(e)
                }
        
        return results