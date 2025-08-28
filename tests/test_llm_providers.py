"""Tests for LLM provider abstraction layer."""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from src.utils.llm.base import ModelTier, LLMResponse
from src.utils.llm.providers import OpenAIProvider, GoogleProvider, OllamaProvider
from src.utils.llm.router import LLMRouter


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
    async def test_router_benchmark(self):
        """Test router benchmarking functionality."""
        config = {
            "ollama_host": "http://192.168.7.43:11434",
            "openai_api_key": None,
            "google_api_key": None
        }
        
        router = LLMRouter(config)
        results = await router.benchmark_providers()
        
        # Should have results for all initialized providers
        assert isinstance(results, dict)
        for provider_name, result in results.items():
            assert "success" in result
            if result["success"]:
                assert "response_time" in result
                assert "model" in result