"""LLM router for intelligent provider selection and fallback handling."""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass

from .base import BaseLLMProvider, LLMResponse, ModelTier
from .providers import OpenAIProvider, GoogleProvider, OllamaProvider
from .metrics import MetricsLogger, PerformanceMonitor, log_llm_request

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
        
        # Model preferences from configuration
        self.heavy_model_preference = config.get("heavy_model_preference", "auto")
        self.light_model_preference = config.get("light_model_preference", "auto")
        
        # Performance monitoring
        self.metrics_logger = MetricsLogger(
            log_file=config.get("metrics_log_file", "logs/llm_metrics.jsonl"),
            enabled=config.get("enable_metrics_logging", True)
        )
        self.performance_monitor = PerformanceMonitor(self.metrics_logger, config)
        
        # Retry configuration
        self.max_retries = config.get("max_retries", 3)
        self.retry_delay = config.get("retry_delay_seconds", 1.0)
        self.provider_timeout = config.get("provider_timeout_seconds", 60)
        
        # Initialize providers based on configuration
        self._initialize_providers()
    
    def _initialize_providers(self) -> None:
        """Initialize all configured providers."""
        provider_configs = {
            "ollama": ProviderConfig(
                provider_class=OllamaProvider,
                config={"host": self.config.get("ollama_host", "http://192.168.1.220:11434")},
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
        """Generate response with intelligent provider selection and retry logic."""
        
        # Select model based on preferences if not specified
        if not model:
            model = self._select_preferred_model(tier, provider)
        
        last_exception = None
        
        # If specific provider is requested, try it first with retries
        if provider and provider in self.providers:
            for attempt in range(self.max_retries):
                try:
                    return await self._generate_with_retry(
                        provider, prompt, tier, model, temperature, max_tokens, **kwargs
                    )
                except Exception as e:
                    last_exception = e
                    logger.warning(f"Requested provider {provider} failed (attempt {attempt + 1}): {e}")
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
        
        # Try providers in fallback order with health-based prioritization
        ordered_providers = self._get_healthy_providers_ordered()
        
        for provider_name in ordered_providers:
            if provider_name not in self.providers:
                continue
                
            for attempt in range(self.max_retries):
                try:
                    return await self._generate_with_retry(
                        provider_name, prompt, tier, model, temperature, max_tokens, **kwargs
                    )
                except Exception as e:
                    last_exception = e
                    logger.warning(f"Provider {provider_name} failed (attempt {attempt + 1}): {e}")
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(self.retry_delay * (2 ** attempt))
                    break  # Move to next provider after max retries
        
        # Log final failure
        log_llm_request(
            provider="all_providers",
            model=model or "unknown",
            tier=tier,
            prompt=prompt,
            response=None,
            error=last_exception
        )
        
        raise RuntimeError(f"All LLM providers failed. Last error: {last_exception}")
    
    async def _generate_with_retry(
        self,
        provider_name: str,
        prompt: str,
        tier: ModelTier,
        model: Optional[str],
        temperature: float,
        max_tokens: Optional[int],
        **kwargs
    ) -> LLMResponse:
        """Generate response with a specific provider with timeout and metrics logging."""
        provider = self.providers[provider_name]
        
        # Select model if not specified
        if not model:
            model = provider.get_model_for_tier(tier)
        
        # Handle GPT-5 specific configuration
        if model.startswith("gpt-5"):
            kwargs.setdefault("reasoning_effort", self.config.get("gpt5_reasoning_effort", "medium"))
            kwargs.setdefault("include_reasoning", self.config.get("gpt5_include_reasoning", False))
            kwargs.setdefault("reasoning_depth", self.config.get("gpt5_reasoning_depth", "standard"))
        
        start_time = time.time()
        response = None
        error = None
        
        try:
            # Apply timeout to provider call
            response = await asyncio.wait_for(
                provider.generate(prompt, model, temperature, max_tokens, **kwargs),
                timeout=self.provider_timeout
            )
            
            # Update provider health on success
            self.provider_health[provider_name] = True
            
            # Log successful request
            log_llm_request(provider_name, model, tier, prompt, response, None)
            
            return response
            
        except asyncio.TimeoutError as e:
            error = TimeoutError(f"Provider {provider_name} timed out after {self.provider_timeout}s")
            self.provider_health[provider_name] = False
            raise error
        except Exception as e:
            error = e
            self.provider_health[provider_name] = False
            raise
        finally:
            # Log request even if it failed
            if response is None:
                log_llm_request(provider_name, model, tier, prompt, None, error)
    
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
    
    def _select_preferred_model(self, tier: ModelTier, provider: Optional[str] = None) -> Optional[str]:
        """Select preferred model based on configuration and availability."""
        preference = self.heavy_model_preference if tier == ModelTier.HEAVY else self.light_model_preference
        
        if preference == "auto":
            return None  # Let provider select default
        
        # Check if preferred model is available in any provider
        for provider_name, provider_instance in self.providers.items():
            if provider and provider_name != provider:
                continue
                
            available_models = provider_instance.get_available_models()
            tier_models = available_models.get(tier, [])
            
            if preference in tier_models:
                return preference
        
        # Fallback to auto selection if preferred model not available
        logger.warning(f"Preferred model {preference} not available for tier {tier}, using auto selection")
        return None
    
    def _get_healthy_providers_ordered(self) -> List[str]:
        """Get providers ordered by health status and fallback order."""
        # Separate healthy and unhealthy providers
        healthy_providers = []
        unhealthy_providers = []
        
        for provider_name in self.fallback_order:
            if provider_name in self.providers:
                if self.provider_health.get(provider_name, True):  # Default to healthy if unknown
                    healthy_providers.append(provider_name)
                else:
                    unhealthy_providers.append(provider_name)
        
        # Return healthy providers first, then unhealthy as last resort
        return healthy_providers + unhealthy_providers
    
    async def start_monitoring(self) -> None:
        """Start performance monitoring."""
        await self.performance_monitor.start_monitoring()
    
    async def stop_monitoring(self) -> None:
        """Stop performance monitoring."""
        await self.performance_monitor.stop_monitoring()
    
    def add_alert_callback(self, callback: callable) -> None:
        """Add callback for performance alerts."""
        self.performance_monitor.add_alert_callback(callback)
    
    def get_metrics_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get comprehensive metrics summary."""
        provider_stats = self.metrics_logger.get_provider_stats(hours)
        model_comparison = self.metrics_logger.get_model_comparison(hours)
        alerts = self.metrics_logger.get_performance_alerts(
            response_time_threshold=self.config.get("performance_threshold_seconds", 30.0),
            cost_threshold=self.config.get("cost_threshold_dollars", 0.10)
        )
        
        return {
            "provider_stats": provider_stats,
            "model_comparison": model_comparison,
            "active_alerts": alerts,
            "provider_health": self.provider_health,
            "configuration": {
                "heavy_model_preference": self.heavy_model_preference,
                "light_model_preference": self.light_model_preference,
                "fallback_order": self.fallback_order,
                "max_retries": self.max_retries,
                "provider_timeout": self.provider_timeout
            }
        }
    
    def export_metrics(self, output_file: str, hours: int = 24, format: str = "json") -> None:
        """Export metrics to file."""
        self.metrics_logger.export_metrics(output_file, hours, format)
    
    async def run_performance_benchmark(
        self,
        test_prompts: Optional[List[str]] = None,
        iterations: int = 3
    ) -> Dict[str, Any]:
        """Run comprehensive performance benchmark across all providers."""
        if test_prompts is None:
            test_prompts = [
                "What is 2+2? Respond with just the number.",
                "Explain the concept of machine learning in one sentence.",
                "List three benefits of renewable energy."
            ]
        
        benchmark_results = {}
        
        for provider_name in self.providers.keys():
            provider_results = {
                "light_tier": [],
                "heavy_tier": []
            }
            
            # Test both tiers
            for tier in [ModelTier.LIGHT, ModelTier.HEAVY]:
                tier_results = []
                
                for prompt in test_prompts:
                    for iteration in range(iterations):
                        try:
                            start_time = time.time()
                            response = await self._generate_with_retry(
                                provider_name, prompt, tier, None, 0.1, 50
                            )
                            
                            tier_results.append({
                                "prompt": prompt[:50] + "..." if len(prompt) > 50 else prompt,
                                "iteration": iteration + 1,
                                "success": True,
                                "response_time": response.response_time,
                                "model": response.model,
                                "tokens": response.tokens_used,
                                "cost": response.cost
                            })
                            
                        except Exception as e:
                            tier_results.append({
                                "prompt": prompt[:50] + "..." if len(prompt) > 50 else prompt,
                                "iteration": iteration + 1,
                                "success": False,
                                "error": str(e)
                            })
                
                provider_results[f"{tier.value}_tier"] = tier_results
            
            benchmark_results[provider_name] = provider_results
        
        return benchmark_results