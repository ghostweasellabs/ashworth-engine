"""OpenAI provider implementation."""

import time
from typing import Dict, Any, Optional, List
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage

from ..base import BaseLLMProvider, LLMResponse, ModelTier


class OpenAIProvider(BaseLLMProvider):
    """OpenAI LLM provider with GPT-5 and GPT-4.1 support."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get("api_key")
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
    
    async def generate(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate response using OpenAI models."""
        start_time = time.time()
        
        try:
            # Handle GPT-5 thinking model with special configuration
            if model.startswith("gpt-5"):
                # GPT-5 requires different prompting strategy
                # See: https://cookbook.openai.com/examples/gpt-5/gpt-5_prompting_guide
                llm = ChatOpenAI(
                    model=model,
                    api_key=self.api_key,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    # GPT-5 specific parameters
                    extra_body={
                        "reasoning_effort": kwargs.get("reasoning_effort", "medium"),
                        "include_reasoning": kwargs.get("include_reasoning", False)
                    }
                )
                
                # Structure prompt for GPT-5 thinking model
                structured_prompt = self._structure_gpt5_prompt(prompt, kwargs)
                messages = [HumanMessage(content=structured_prompt)]
            else:
                # Standard GPT-4.1 models
                llm = ChatOpenAI(
                    model=model,
                    api_key=self.api_key,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                messages = [HumanMessage(content=prompt)]
            
            response = await llm.ainvoke(messages)
            response_time = time.time() - start_time
            
            return LLMResponse(
                content=response.content,
                model=model,
                provider="openai",
                tokens_used=response.response_metadata.get("token_usage", {}).get("total_tokens"),
                response_time=response_time,
                cost=self._calculate_cost(model, response.response_metadata.get("token_usage", {}))
            )
            
        except Exception as e:
            raise RuntimeError(f"OpenAI generation failed: {str(e)}")
    
    def _structure_gpt5_prompt(self, prompt: str, kwargs: Dict[str, Any]) -> str:
        """Structure prompt for GPT-5 thinking model."""
        reasoning_depth = kwargs.get("reasoning_depth", "standard")
        
        if reasoning_depth == "thorough":
            prefix = "Think through this step by step, considering multiple perspectives and potential edge cases:\n\n"
        elif reasoning_depth == "quick":
            prefix = "Provide a direct, concise response:\n\n"
        else:  # standard
            prefix = "Analyze this carefully and provide a well-reasoned response:\n\n"
        
        return f"{prefix}{prompt}"
    
    def _calculate_cost(self, model: str, token_usage: Dict[str, Any]) -> Optional[float]:
        """Calculate approximate cost based on token usage."""
        if not token_usage:
            return None
            
        # Approximate pricing (as of 2025)
        pricing = {
            "gpt-5": {"input": 0.00003, "output": 0.00012},  # $30/$120 per 1M tokens
            "gpt-4.1": {"input": 0.000015, "output": 0.00006},  # $15/$60 per 1M tokens
            "gpt-4.1-mini": {"input": 0.000001, "output": 0.000004},  # $1/$4 per 1M tokens
        }
        
        model_key = model.split("-")[0] + "-" + model.split("-")[1]
        if model_key not in pricing:
            return None
            
        input_tokens = token_usage.get("prompt_tokens", 0)
        output_tokens = token_usage.get("completion_tokens", 0)
        
        cost = (input_tokens * pricing[model_key]["input"] + 
                output_tokens * pricing[model_key]["output"])
        return cost
    
    def get_available_models(self) -> Dict[ModelTier, List[str]]:
        """Get available OpenAI models by tier."""
        return {
            ModelTier.HEAVY: ["gpt-5", "gpt-4.1"],
            ModelTier.LIGHT: ["gpt-4.1-mini", "gpt-4.1"]
        }
    
    async def health_check(self) -> bool:
        """Check OpenAI API availability."""
        try:
            llm = ChatOpenAI(
                model="gpt-4.1-mini",
                api_key=self.api_key,
                max_tokens=10
            )
            await llm.ainvoke([HumanMessage(content="test")])
            return True
        except Exception:
            return False