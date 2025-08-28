"""Google AI provider implementation."""

import time
from typing import Dict, Any, Optional, List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage

from ..base import BaseLLMProvider, LLMResponse, ModelTier


class GoogleProvider(BaseLLMProvider):
    """Google AI provider with Gemini 2.5 support."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get("api_key")
        if not self.api_key:
            raise ValueError("Google API key is required")
    
    async def generate(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate response using Google Gemini models."""
        start_time = time.time()
        
        try:
            llm = ChatGoogleGenerativeAI(
                model=model,
                google_api_key=self.api_key,
                temperature=temperature,
                max_output_tokens=max_tokens
            )
            
            messages = [HumanMessage(content=prompt)]
            response = await llm.ainvoke(messages)
            response_time = time.time() - start_time
            
            return LLMResponse(
                content=response.content,
                model=model,
                provider="google",
                tokens_used=response.response_metadata.get("usage_metadata", {}).get("total_token_count"),
                response_time=response_time,
                cost=self._calculate_cost(model, response.response_metadata.get("usage_metadata", {}))
            )
            
        except Exception as e:
            raise RuntimeError(f"Google AI generation failed: {str(e)}")
    
    def _calculate_cost(self, model: str, usage_metadata: Dict[str, Any]) -> Optional[float]:
        """Calculate approximate cost based on token usage."""
        if not usage_metadata:
            return None
            
        # Approximate Gemini pricing (as of 2025)
        pricing = {
            "gemini-2.5-pro": {"input": 0.000007, "output": 0.000021},  # $7/$21 per 1M tokens
            "gemini-2.5-flash": {"input": 0.0000002, "output": 0.0000006},  # $0.2/$0.6 per 1M tokens
        }
        
        if model not in pricing:
            return None
            
        input_tokens = usage_metadata.get("prompt_token_count", 0)
        output_tokens = usage_metadata.get("candidates_token_count", 0)
        
        cost = (input_tokens * pricing[model]["input"] + 
                output_tokens * pricing[model]["output"])
        return cost
    
    def get_available_models(self) -> Dict[ModelTier, List[str]]:
        """Get available Google models by tier."""
        return {
            ModelTier.HEAVY: ["gemini-2.5-pro"],
            ModelTier.LIGHT: ["gemini-2.5-flash"]
        }
    
    async def health_check(self) -> bool:
        """Check Google AI API availability."""
        try:
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                google_api_key=self.api_key,
                max_output_tokens=10
            )
            await llm.ainvoke([HumanMessage(content="test")])
            return True
        except Exception:
            return False