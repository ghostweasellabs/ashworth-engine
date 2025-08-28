"""Ollama provider implementation for local models."""

import time
from typing import Dict, Any, Optional, List
import ollama
from langchain.schema import HumanMessage

from ..base import BaseLLMProvider, LLMResponse, ModelTier


class OllamaProvider(BaseLLMProvider):
    """Ollama provider for local model execution."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.host = config.get("host", "http://localhost:11434")
        self.client = ollama.Client(host=self.host)
    
    async def generate(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate response using Ollama models."""
        start_time = time.time()
        
        try:
            # Ollama generate call
            response = self.client.generate(
                model=model,
                prompt=prompt,
                options={
                    "temperature": temperature,
                    "num_predict": max_tokens or -1,  # -1 means no limit
                }
            )
            
            response_time = time.time() - start_time
            
            return LLMResponse(
                content=response["response"],
                model=model,
                provider="ollama",
                tokens_used=response.get("eval_count", 0) + response.get("prompt_eval_count", 0),
                response_time=response_time,
                cost=0.0  # Local models have no API cost
            )
            
        except Exception as e:
            raise RuntimeError(f"Ollama generation failed: {str(e)}")
    
    def get_available_models(self) -> Dict[ModelTier, List[str]]:
        """Get available Ollama models by tier."""
        try:
            # Get list of available models from Ollama
            models_response = self.client.list()
            available_models = [model["name"] for model in models_response.get("models", [])]
            
            # Categorize models based on naming patterns and known capabilities
            heavy_models = []
            light_models = []
            
            for model in available_models:
                model_lower = model.lower()
                # Heavy models: larger parameter counts, specialized reasoning models
                if any(pattern in model_lower for pattern in ["gpt-oss", "llama3", "mixtral", "qwen", "70b", "34b"]):
                    heavy_models.append(model)
                # Light models: smaller, efficient models
                elif any(pattern in model_lower for pattern in ["gemma", "phi", "7b", "8b", "mini", "small"]):
                    light_models.append(model)
                else:
                    # Default to light for unknown models
                    light_models.append(model)
            
            return {
                ModelTier.HEAVY: heavy_models,
                ModelTier.LIGHT: light_models
            }
            
        except Exception:
            # Fallback to expected models if we can't query Ollama
            return {
                ModelTier.HEAVY: ["gpt-oss:20b", "llama3:70b"],
                ModelTier.LIGHT: ["gemma3n:e4b", "phi3:mini"]
            }
    
    async def health_check(self) -> bool:
        """Check Ollama server availability."""
        try:
            # Try to list models to verify connection
            self.client.list()
            return True
        except Exception:
            return False
    
    def get_model_info(self, model: str) -> Dict[str, Any]:
        """Get detailed information about a specific model."""
        try:
            return self.client.show(model)
        except Exception:
            return {}