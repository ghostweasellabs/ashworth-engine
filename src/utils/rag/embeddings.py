"""
Embedding generation using configurable models (OpenAI, Ollama, etc.).
Handles rate limiting, retries, and batch processing.
"""

import asyncio
import logging
from typing import List, Optional, Dict, Any

import openai
from openai import AsyncOpenAI
import httpx

from src.config.settings import get_settings

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generates embeddings using configurable models (OpenAI, Ollama, etc.)."""

    def __init__(self):
        self.settings = get_settings()
        self.model = self.settings.embedding_model or "embeddinggemma"
        self.provider = self._determine_provider()
        self.embedding_dimension = self._get_embedding_dimension()

        # Initialize appropriate client based on provider
        if self.provider == "openai":
            self.client = AsyncOpenAI(api_key=self.settings.openai_api_key)
        elif self.provider == "ollama":
            self.client = httpx.AsyncClient(base_url=self.settings.ollama_host)
        else:
            raise ValueError(f"Unsupported embedding provider: {self.provider}")

    def _determine_provider(self) -> str:
        """Determine which provider to use based on the model."""
        if self.model == "text-embedding-ada-002":
            return "openai"
        elif self.model == "embeddinggemma":
            return "ollama"
        else:
            # Default to ollama for unknown models
            return "ollama"

    def _get_embedding_dimension(self) -> int:
        """Get the embedding dimension for the current model."""
        dimension_map = {
            "text-embedding-ada-002": 1536,
            "embeddinggemma": 3584,  # Gemma embedding dimension
        }
        return dimension_map.get(self.model, 1536)  # Default fallback
        
    async def generate_embedding(self, text: str) -> Optional[List[float]]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            List of floats representing the embedding, or None if failed
        """
        try:
            # Clean and truncate text if needed
            cleaned_text = self._clean_text(text)

            if self.provider == "openai":
                return await self._generate_openai_embedding(cleaned_text)
            elif self.provider == "ollama":
                return await self._generate_ollama_embedding(cleaned_text)
            else:
                logger.error(f"Unsupported provider: {self.provider}")
                return None

        except Exception as e:
            logger.error(f"Unexpected error generating embedding: {e}")
            return None

    async def _generate_openai_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding using OpenAI API."""
        try:
            response = await self.client.embeddings.create(
                model=self.model,
                input=text
            )

            embedding = response.data[0].embedding

            # Validate embedding dimension
            if len(embedding) != self.embedding_dimension:
                logger.error(f"Unexpected embedding dimension: {len(embedding)}")
                return None

            return embedding

        except openai.RateLimitError as e:
            logger.warning(f"Rate limit hit, retrying after delay: {e}")
            await asyncio.sleep(60)  # Wait 1 minute
            return await self._generate_openai_embedding(text)

        except openai.APIError as e:
            logger.error(f"OpenAI API error: {e}")
            return None

    async def _generate_ollama_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding using Ollama API."""
        try:
            # Ollama embedding API format
            payload = {
                "model": self.model,
                "prompt": text
            }

            response = await self.client.post("/api/embeddings", json=payload, timeout=30.0)
            
            if response.status_code != 200:
                logger.error(f"Ollama API error: {response.status_code} - {response.text}")
                return None
                
            response_data = response.json()

            if "embedding" in response_data:
                embedding = response_data["embedding"]

                # Validate embedding dimension
                if len(embedding) != self.embedding_dimension:
                    logger.warning(f"Unexpected embedding dimension: {len(embedding)}, expected {self.embedding_dimension}")

                return embedding
            else:
                logger.error(f"No embedding in Ollama response: {response_data}")
                return None

        except httpx.HTTPStatusError as e:
            logger.error(f"Ollama HTTP error: {e.response.status_code} - {e.response.text}")
            return None
        except httpx.TimeoutException:
            logger.error("Ollama embedding request timed out")
            return None
        except Exception as e:
            logger.error(f"Ollama API error: {e}")
            return None
    
    async def generate_embeddings_batch(
        self, 
        texts: List[str], 
        batch_size: int = 100
    ) -> List[Optional[List[float]]]:
        """
        Generate embeddings for multiple texts in batches.
        
        Args:
            texts: List of texts to embed
            batch_size: Number of texts to process in each batch
            
        Returns:
            List of embeddings (same order as input texts)
        """
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            logger.info(f"Processing embedding batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
            
            batch_embeddings = await self._process_batch(batch)
            embeddings.extend(batch_embeddings)
            
            # Add delay between batches to respect rate limits
            if i + batch_size < len(texts):
                await asyncio.sleep(1)
        
        return embeddings
    
    async def _process_batch(self, texts: List[str]) -> List[Optional[List[float]]]:
        """Process a batch of texts concurrently."""
        tasks = [self.generate_embedding(text) for text in texts]
        return await asyncio.gather(*tasks, return_exceptions=False)
    
    def _clean_text(self, text: str) -> str:
        """Clean and prepare text for embedding."""
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Truncate if too long (OpenAI has token limits)
        max_chars = 8000  # Conservative limit
        if len(text) > max_chars:
            text = text[:max_chars]
            logger.warning(f"Text truncated to {max_chars} characters")
        
        return text
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this generator."""
        return self.embedding_dimension


# Convenience function for testing
async def test_embedding_generation():
    """Test embedding generation with sample text."""
    generator = EmbeddingGenerator()
    
    sample_text = "Business expenses must be ordinary and necessary to be deductible."
    embedding = await generator.generate_embedding(sample_text)
    
    if embedding:
        print(f"Generated embedding with dimension: {len(embedding)}")
        print(f"First 5 values: {embedding[:5]}")
        return True
    else:
        print("Failed to generate embedding")
        return False


if __name__ == "__main__":
    asyncio.run(test_embedding_generation())