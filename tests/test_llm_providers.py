"""Tests for LLM provider abstraction layer."""

import pytest
import asyncio
import tempfile
import os
from unittest.mock import Mock, patch, AsyncMock
from src.utils.llm.base import ModelTier, LLMResponse
from src.utils.llm.providers import OpenAIProvider, GoogleProvider, OllamaProvider
from src.utils.llm.router import LLMRouter
from src.utils.llm.metrics import MetricsLogger, MetricsEntry
from src.utils.llm.config_manager import LLMConfigManager


class TestBaseLLMProvider:
    """Test base LLM provider functionality."""
    
    def test_model_tier_enum(self):
        """Test ModelTier enum values."""
        assert ModelTier.HEAVY.value == "heavy"
        assert ModelTier.LIGHT.value == "light"
    
    def test_llm_response_dataclass(self):
        """Test LLMResponse dataclass."""
        response = LLMResponse(
            content="Test response",
            model="test-model",
            provider="test-provider"
        )
        
        assert response.content == "Test response"
        assert response.model == "test-model"
        assert response.provider == "test-provider"
        assert response.timestamp is not None


class TestOpenAIProvider:
    """Test OpenAI provider implementation."""
    
    def test_init_without_api_key(self):
        """Test initialization fails without API key."""
        with pytest.raises(ValueError, match="OpenAI API key is required"):
            OpenAIProvider({})
    
    def test_init_with_api_key(self):
        """Test successful initialization with API key."""
        provider = OpenAIProvider({"api_key": "test-key"})
        assert provider.api_key == "test-key"
        assert provider.name == "openai"
    
    def test_get_available_models(self):
        """Test getting available models by tier."""
        provider = OpenAIProvider({"api_key": "test-key"})
        models = provider.get_available_models()
        
        assert ModelTier.HEAVY in models
        assert ModelTier.LIGHT in models
        assert "gpt-5" in models[ModelTier.HEAVY]
        assert "gpt-4.1-mini" in models[ModelTier.LIGHT]
    
    def test_gpt5_prompt_structuring(self):
        """Test GPT-5 prompt structuring."""
        provider = OpenAIProvider({"api_key": "test-key"})
        
        # Test thorough reasoning
        prompt = provider._structure_gpt5_prompt("Test prompt", {"reasoning_depth": "thorough"})
        assert "step by step" in prompt
        assert "Test prompt" in prompt
        
        # Test quick reasoning
        prompt = provider._structure_gpt5_prompt("Test prompt", {"reasoning_depth": "quick"})
        assert "direct, concise" in prompt
        
        # Test standard reasoning
        prompt = provider._structure_gpt5_prompt("Test prompt", {})
        assert "well-reasoned" in prompt
    
    @pytest.mark.asyncio
    async def test_health_check_failure(self):
        """Test health check with invalid API key."""
        provider = OpenAIProvider({"api_key": "invalid-key"})
        health = await provider.health_check()
        assert health is False


class TestGoogleProvider:
    """Test Google provider implementation."""
    
    def test_init_without_api_key(self):
        """Test initialization fails without API key."""
        with pytest.raises(ValueError, match="Google API key is required"):
            GoogleProvider({})
    
    def test_init_with_api_key(self):
        """Test successful initialization with API key."""
        provider = GoogleProvider({"api_key": "test-key"})
        assert provider.api_key == "test-key"
        assert provider.name == "google"
    
    def test_get_available_models(self):
        """Test getting available models by tier."""
        provider = GoogleProvider({"api_key": "test-key"})
        models = provider.get_available_models()
        
        assert ModelTier.HEAVY in models
        assert ModelTier.LIGHT in models
        assert "gemini-2.5-pro" in models[ModelTier.HEAVY]
        assert "gemini-2.5-flash" in models[ModelTier.LIGHT]


class TestOllamaProvider:
    """Test Ollama provider implementation."""
    
    def test_init_default_host(self):
        """Test initialization with default host."""
        provider = OllamaProvider({})
        assert provider.host == "http://localhost:11434"
        assert provider.name == "ollama"
    
    def test_init_custom_host(self):
        """Test initialization with custom host."""
        provider = OllamaProvider({"host": "http://192.168.7.43:11434"})
        assert provider.host == "http://192.168.7.43:11434"
    
    @patch('ollama.Client')
    def test_get_available_models_success(self, mock_client):
        """Test getting available models from Ollama."""
        # Mock Ollama client response
        mock_instance = Mock()
        mock_instance.list.return_value = {
            "models": [
                {"name": "gpt-oss:20b"},
                {"name": "gemma3n:e4b"},
                {"name": "llama3:70b"},
                {"name": "phi3:mini"}
            ]
        }
        mock_client.return_value = mock_instance
        
        provider = OllamaProvider({})
        models = provider.get_available_models()
        
        assert ModelTier.HEAVY in models
        assert ModelTier.LIGHT in models
        assert "gpt-oss:20b" in models[ModelTier.HEAVY]
        assert "gemma3n:e4b" in models[ModelTier.LIGHT]
    
    @patch('ollama.Client')
    def test_get_available_models_fallback(self, mock_client):
        """Test fallback when Ollama query fails."""
        mock_instance = Mock()
        mock_instance.list.side_effect = Exception("Connection failed")
        mock_client.return_value = mock_instance
        
        provider = OllamaProvider({})
        models = provider.get_available_models()
        
        # Should return fallback models
        assert "gpt-oss:20b" in models[ModelTier.HEAVY]
        assert "gemma3n:e4b" in models[ModelTier.LIGHT]


class TestLLMRouter:
    """Test LLM router functionality."""
    
    def test_init_with_config(self):
        """Test router initialization."""
        config = {
            "ollama_host": "http://test:11434",
            "openai_api_key": "test-openai-key",
            "google_api_key": "test-google-key",
            "fallback_order": ["ollama", "openai", "google"]
        }
        
        router = LLMRouter(config)
        assert router.fallback_order == ["ollama", "openai", "google"]
        assert len(router.providers) >= 1  # At least Ollama should be initialized
    
    @pytest.mark.asyncio
    async def test_generate_with_fallback(self):
        """Test generation with provider fallback."""
        config = {
            "ollama_host": "http://invalid:11434",  # This will fail
            "openai_api_key": None,  # No API key
            "google_api_key": None,  # No API key
            "fallback_order": ["ollama", "openai", "google"]
        }
        
        router = LLMRouter(config)
        
        # All providers should fail, resulting in RuntimeError
        with pytest.raises(RuntimeError, match="All LLM providers failed"):
            await router.generate("test prompt", ModelTier.LIGHT)
    
    @pytest.mark.asyncio
    async def test_health_check_all(self):
        """Test health check for all providers."""
        config = {
            "ollama_host": "http://invalid:11434",
            "openai_api_key": None,
            "google_api_key": None
        }
        
        router = LLMRouter(config)
        health_results = await router.health_check_all()
        
        # All should be False due to invalid configs
        for provider_name, health in health_results.items():
            assert health is False
    
    def test_get_available_models(self):
        """Test getting models from all providers."""
        config = {
            "ollama_host": "http://test:11434",
            "openai_api_key": "test-key",
            "google_api_key": "test-key"
        }
        
        router = LLMRouter(config)
        all_models = router.get_available_models()
        
        # Should have models from all initialized providers
        assert isinstance(all_models, dict)
        for provider_models in all_models.values():
            assert ModelTier.HEAVY in provider_models
            assert ModelTier.LIGHT in provider_models
    
    def test_get_provider_status(self):
        """Test getting provider status information."""
        config = {
            "ollama_host": "http://test:11434",
            "openai_api_key": "test-key",
            "google_api_key": None  # Disabled
        }
        
        router = LLMRouter(config)
        status = router.get_provider_status()
        
        assert isinstance(status, dict)
        for provider_name, provider_status in status.items():
            assert "available" in provider_status
            assert "models" in provider_status
            assert "config" in provider_status


class TestMetricsLogger:
    """Test metrics logging functionality."""
    
    def test_metrics_entry_creation(self):
        """Test creating metrics entry."""
        from datetime import datetime
        
        entry = MetricsEntry(
            timestamp=datetime.utcnow(),
            provider="test",
            model="test-model",
            tier="light",
            prompt_length=100,
            response_length=50,
            tokens_used=75,
            response_time=1.5,
            cost=0.001,
            success=True
        )
        
        assert entry.provider == "test"
        assert entry.success is True
        assert entry.response_time == 1.5
    
    def test_metrics_logger_initialization(self):
        """Test metrics logger initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = os.path.join(temp_dir, "test_metrics.jsonl")
            logger = MetricsLogger(log_file=log_file, enabled=True)
            
            assert logger.enabled is True
            assert logger.log_file.name == "test_metrics.jsonl"
    
    def test_log_request_success(self):
        """Test logging successful request."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = os.path.join(temp_dir, "test_metrics.jsonl")
            logger = MetricsLogger(log_file=log_file, enabled=True)
            
            response = LLMResponse(
                content="Test response",
                model="test-model",
                provider="test",
                tokens_used=50,
                response_time=1.0,
                cost=0.001
            )
            
            logger.log_request("test", "test-model", ModelTier.LIGHT, "test prompt", response)
            
            # Check that entry was added to in-memory storage
            assert len(logger.in_memory_metrics) == 1
            entry = logger.in_memory_metrics[0]
            assert entry.success is True
            assert entry.provider == "test"
    
    def test_log_request_failure(self):
        """Test logging failed request."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = os.path.join(temp_dir, "test_metrics.jsonl")
            logger = MetricsLogger(log_file=log_file, enabled=True)
            
            error = Exception("Test error")
            logger.log_request("test", "test-model", ModelTier.LIGHT, "test prompt", None, error)
            
            # Check that entry was added to in-memory storage
            assert len(logger.in_memory_metrics) == 1
            entry = logger.in_memory_metrics[0]
            assert entry.success is False
            assert entry.error_message == "Test error"


class TestLLMConfigManager:
    """Test LLM configuration management."""
    
    def test_config_manager_initialization(self):
        """Test config manager initialization."""
        config_manager = LLMConfigManager()
        
        assert len(config_manager.models) > 0
        assert "gpt-5" in config_manager.models
        assert "gemini-2.5-pro" in config_manager.models
        assert "gpt-oss:20b" in config_manager.models
    
    def test_get_models_by_tier(self):
        """Test getting models by tier."""
        config_manager = LLMConfigManager()
        
        heavy_models = config_manager.get_models_by_tier(ModelTier.HEAVY)
        light_models = config_manager.get_models_by_tier(ModelTier.LIGHT)
        
        assert len(heavy_models) > 0
        assert len(light_models) > 0
        
        # Check that all heavy models are actually heavy tier
        for model in heavy_models:
            assert model.tier == ModelTier.HEAVY
    
    def test_get_router_config(self):
        """Test getting router configuration."""
        config_manager = LLMConfigManager()
        config = config_manager.get_router_config()
        
        assert "ollama_host" in config
        assert "fallback_order" in config
        assert "enable_metrics_logging" in config
        assert isinstance(config["fallback_order"], list)
    
    def test_validate_configuration(self):
        """Test configuration validation."""
        config_manager = LLMConfigManager()
        validation = config_manager.validate_configuration()
        
        assert "valid" in validation
        assert "warnings" in validation
        assert "errors" in validation
        assert "provider_status" in validation


class TestEnhancedLLMRouter:
    """Test enhanced LLM router with monitoring."""
    
    def test_router_with_metrics(self):
        """Test router initialization with metrics."""
        config = {
            "ollama_host": "http://test:11434",
            "enable_metrics_logging": True,
            "metrics_log_file": "test_metrics.jsonl",
            "max_retries": 2,
            "retry_delay_seconds": 0.1
        }
        
        router = LLMRouter(config)
        
        assert router.metrics_logger is not None
        assert router.performance_monitor is not None
        assert router.max_retries == 2
        assert router.retry_delay == 0.1
    
    def test_model_preference_selection(self):
        """Test model preference selection."""
        config = {
            "ollama_host": "http://test:11434",
            "heavy_model_preference": "gpt-5",
            "light_model_preference": "gpt-4.1-mini"
        }
        
        router = LLMRouter(config)
        
        assert router.heavy_model_preference == "gpt-5"
        assert router.light_model_preference == "gpt-4.1-mini"
    
    def test_get_metrics_summary(self):
        """Test getting metrics summary."""
        config = {
            "ollama_host": "http://test:11434",
            "enable_metrics_logging": True
        }
        
        router = LLMRouter(config)
        summary = router.get_metrics_summary()
        
        assert "provider_stats" in summary
        assert "model_comparison" in summary
        assert "active_alerts" in summary
        assert "provider_health" in summary
        assert "configuration" in summary


@pytest.mark.integration
class TestLLMIntegration:
    """Integration tests for LLM providers (requires actual services)."""
    
    @pytest.mark.asyncio
    async def test_ollama_integration(self):
        """Test Ollama integration if available."""
        provider = OllamaProvider({"host": "http://192.168.7.43:11434"})
        
        try:
            health = await provider.health_check()
            if health:
                # Only run if Ollama is actually available
                models = provider.get_available_models()
                assert len(models[ModelTier.HEAVY]) > 0 or len(models[ModelTier.LIGHT]) > 0
                
                # Test simple generation if models are available
                if models[ModelTier.LIGHT]:
                    response = await provider.generate(
                        "What is 2+2?",
                        models[ModelTier.LIGHT][0],
                        temperature=0.1,
                        max_tokens=10
                    )
                    assert response.content
                    assert response.provider == "ollama"
                    assert response.cost == 0.0  # Local models are free
        except Exception:
            pytest.skip("Ollama not available for integration test")
    
    @pytest.mark.asyncio
    async def test_enhanced_router_with_retries(self):
        """Test enhanced router with retry logic."""
        config = {
            "ollama_host": "http://192.168.7.43:11434",
            "openai_api_key": None,
            "google_api_key": None,
            "max_retries": 2,
            "retry_delay_seconds": 0.1,
            "enable_metrics_logging": True
        }
        
        router = LLMRouter(config)
        
        try:
            # This should work if Ollama is available
            response = await router.generate("What is 2+2?", ModelTier.LIGHT)
            assert response.content
            
            # Check that metrics were logged
            summary = router.get_metrics_summary(hours=1)
            assert "provider_stats" in summary
            
        except RuntimeError as e:
            # Expected if no providers are available
            assert "All LLM providers failed" in str(e)
    
    @pytest.mark.asyncio
    async def test_performance_benchmark(self):
        """Test performance benchmarking."""
        config = {
            "ollama_host": "http://192.168.7.43:11434",
            "enable_metrics_logging": True
        }
        
        router = LLMRouter(config)
        
        try:
            results = await router.run_performance_benchmark(
                test_prompts=["What is 2+2?"],
                iterations=1
            )
            
            assert isinstance(results, dict)
            # Should have results for at least one provider
            assert len(results) > 0
            
        except Exception:
            pytest.skip("No providers available for benchmark test")