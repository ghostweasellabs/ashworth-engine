"""LLM integration utilities for OpenAI and Anthropic APIs"""

from typing import Dict, List, Any, Optional
from src.config.settings import settings
from openai import OpenAI
import asyncio
from functools import wraps

class LLMClient:
    """LLM client with model routing for optimal performance and cost"""
    
    def __init__(self, provider: str = None):
        self.provider = provider or settings.llm_provider
        if self.provider == "openai":
            self.client = OpenAI(api_key=settings.openai_api_key)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def _select_model(self, task_type: str = "general") -> str:
        """Select appropriate model based on task complexity"""
        if task_type in ["classification", "categorization", "simple_analysis"]:
            if settings.use_mini_for_classification:
                return settings.secondary_model
        elif task_type in ["report_generation", "complex_analysis", "narrative"]:
            if settings.use_primary_for_reports:
                return settings.primary_model
        
        # Default to primary model
        return settings.primary_model
    
    def generate_text(self, prompt: str, task_type: str = "general", **kwargs) -> str:
        """Generate text using configured LLM provider"""
        if self.provider != "openai":
            raise NotImplementedError(f"Provider {self.provider} not implemented")
        
        model = self._select_model(task_type)
        max_tokens = kwargs.get('max_tokens', 
            settings.max_tokens_primary if model == settings.primary_model 
            else settings.max_tokens_secondary
        )
        
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a financial analysis expert."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=kwargs.get('temperature', settings.temperature)
            )
            return response.choices[0].message.content
        except Exception as e:
            # Fallback to secondary model if primary fails
            if model == settings.primary_model:
                try:
                    response = self.client.chat.completions.create(
                        model=settings.secondary_model,
                        messages=[
                            {"role": "system", "content": "You are a financial analysis expert."},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=settings.max_tokens_secondary,
                        temperature=kwargs.get('temperature', settings.temperature)
                    )
                    return response.choices[0].message.content
                except Exception:
                    pass
            raise e
    
    async def generate_text_async(self, prompt: str, task_type: str = "general", **kwargs) -> str:
        """Async wrapper for generate_text"""
        return await asyncio.get_event_loop().run_in_executor(
            None, self.generate_text, prompt, task_type, **kwargs
        )
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for texts"""
        if self.provider != "openai":
            raise NotImplementedError(f"Provider {self.provider} not implemented")
        
        try:
            response = self.client.embeddings.create(
                model=settings.embedding_model,
                input=texts
            )
            return [embedding.embedding for embedding in response.data]
        except Exception as e:
            raise e

# Global LLM client instance
llm_client = LLMClient()