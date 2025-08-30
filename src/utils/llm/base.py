"""Base LLM provider interface and common types."""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime


class ModelTier(Enum):
    """Model capability tiers for routing decisions."""
    HEAVY = "heavy"  # Complex reasoning, report generation
    LIGHT = "light"  # Simple categorization, validation


@dataclass
class LLMResponse:
    """Standardized response from LLM providers."""
    content: str
    model: str
    provider: str
    tokens_used: Optional[int] = None
    response_time: Optional[float] = None
    cost: Optional[float] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize provider with configuration."""
        self.config = config
        self.name = self.__class__.__name__.replace("Provider", "").lower()
        
    @abstractmethod
    async def generate(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate response from the LLM."""
        pass
    
    @abstractmethod
    def get_available_models(self) -> Dict[ModelTier, List[str]]:
        """Get available models organized by capability tier."""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the provider is available and healthy."""
        pass
    
    def get_model_for_tier(self, tier: ModelTier) -> str:
        """Get the default model for a given tier."""
        models = self.get_available_models()
        if tier in models and models[tier]:
            return models[tier][0]  # Return first available model
        raise ValueError(f"No models available for tier {tier}")