"""
Configurable similarity search with multiple search strategies and thresholds.
"""

import logging
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from src.utils.rag.embeddings import EmbeddingGenerator
from src.utils.rag.vector_store import SearchResult, VectorStore

logger = logging.getLogger(__name__)


class SearchStrategy(str, Enum):
    """Different search strategies for similarity search."""
    STRICT = "strict"  # High threshold, fewer but more relevant results
    BALANCED = "balanced"  # Medium threshold, balanced relevance/recall
    BROAD = "broad"  # Lower threshold, more results with potential noise
    ADAPTIVE = "adaptive"  # Dynamically adjust threshold based on results


class SearchConfig(BaseModel):
    """Configuration for similarity search."""
    strategy: SearchStrategy = SearchStrategy.BALANCED
    max_results: int = Field(default=10, ge=1, le=100)
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    document_type_filter: Optional[str] = None
    metadata_filters: Dict = Field(default_factory=dict)
    rerank_results: bool = True
    include_context: bool = True  # Include surrounding chunks for context


class SimilaritySearchEngine:
    """Advanced similarity search engine with configurable strategies."""
    
    def __init__(self):
        self.embedding_generator = EmbeddingGenerator()
        self.vector_store = VectorStore()
        
        # Strategy-specific thresholds
        self.strategy_thresholds = {
            SearchStrategy.STRICT: 0.85,
            SearchStrategy.BALANCED: 0.7,
            SearchStrategy.BROAD: 0.5,
            SearchStrategy.ADAPTIVE: 0.7  # Starting point for adaptive
        }
    
    async def search(self, query: str, config: SearchConfig = None) -> List[SearchResult]:
        """
        Perform similarity search with the given configuration.
        
        Args:
            query: Search query text
            config: Search configuration
            
        Returns:
            List of search results
        """
        if config is None:
            config = SearchConfig()
        
        try:
            logger.info(f"Performing {config.strategy} search for query: {query[:100]}...")
            
            # Generate query embedding
            query_embedding = await self.embedding_generator.generate_embedding(query)
            if not query_embedding:
                logger.error("Failed to generate query embedding")
                return []
            
            # Get threshold for strategy
            threshold = self._get_threshold_for_strategy(config)
            
            # Perform initial search
            results = await self.vector_store.similarity_search(
                query_embedding=query_embedding,
                limit=config.max_results * 2,  # Get more results for potential filtering
                similarity_threshold=threshold,
                document_type=config.document_type_filter,
                metadata_filter=config.metadata_filters
            )
            
            # Apply strategy-specific processing
            results = await self._apply_strategy_processing(results, config, query)
            
            # Limit to requested number of results
            results = results[:config.max_results]
            
            logger.info(f"Returning {len(results)} search results")
            return results
            
        except Exception as e:
            logger.error(f"Error performing similarity search: {e}")
            return []
    
    def _get_threshold_for_strategy(self, config: SearchConfig) -> float:
        """Get similarity threshold based on strategy."""
        if config.similarity_threshold != 0.7:  # User specified custom threshold
            return config.similarity_threshold
        
        return self.strategy_thresholds.get(config.strategy, 0.7)
    
    async def _apply_strategy_processing(
        self, 
        results: List[SearchResult], 
        config: SearchConfig, 
        query: str
    ) -> List[SearchResult]:
        """Apply strategy-specific processing to results."""
        
        if config.strategy == SearchStrategy.STRICT:
            # Keep only highest quality results
            results = [r for r in results if r.similarity_score >= 0.85]
            
        elif config.strategy == SearchStrategy.ADAPTIVE:
            # Adjust threshold based on result distribution
            if results:
                scores = [r.similarity_score for r in results]
                avg_score = sum(scores) / len(scores)
                
                # If average score is high, be more selective
                if avg_score > 0.8:
                    adaptive_threshold = 0.75
                else:
                    adaptive_threshold = 0.6
                
                results = [r for r in results if r.similarity_score >= adaptive_threshold]
        
        # Rerank results if requested
        if config.rerank_results:
            results = await self._rerank_results(results, query)
        
        # Add context if requested
        if config.include_context:
            results = await self._add_context_to_results(results)
        
        return results
    
    async def _rerank_results(self, results: List[SearchResult], query: str) -> List[SearchResult]:
        """Rerank results based on additional relevance signals."""
        # Simple reranking based on content length and keyword matching
        query_words = set(query.lower().split())
        
        for result in results:
            content_words = set(result.content.lower().split())
            keyword_overlap = len(query_words.intersection(content_words)) / len(query_words)
            
            # Adjust score based on keyword overlap and content quality
            content_quality = min(len(result.content) / 500, 1.0)  # Prefer substantial content
            
            # Combine similarity score with additional signals
            result.similarity_score = (
                result.similarity_score * 0.7 +  # Original similarity
                keyword_overlap * 0.2 +          # Keyword overlap
                content_quality * 0.1            # Content quality
            )
        
        # Re-sort by adjusted scores
        results.sort(key=lambda x: x.similarity_score, reverse=True)
        return results
    
    async def _add_context_to_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """Add surrounding chunks as context to search results."""
        # Group results by document
        doc_results = {}
        for result in results:
            if result.document_id not in doc_results:
                doc_results[result.document_id] = []
            doc_results[result.document_id].append(result)
        
        # For each document, get surrounding chunks
        enhanced_results = []
        for doc_id, doc_results_list in doc_results.items():
            doc_chunks = await self.vector_store.get_document_chunks(doc_id)
            
            for result in doc_results_list:
                # Add context from surrounding chunks
                context_before = ""
                context_after = ""
                
                # Find chunks before and after current chunk
                for chunk in doc_chunks:
                    if chunk['chunk_index'] == result.chunk_index - 1:
                        context_before = chunk['content'][-200:]  # Last 200 chars
                    elif chunk['chunk_index'] == result.chunk_index + 1:
                        context_after = chunk['content'][:200]   # First 200 chars
                
                # Add context to metadata
                if context_before or context_after:
                    result.metadata['context_before'] = context_before
                    result.metadata['context_after'] = context_after
                
                enhanced_results.append(result)
        
        return enhanced_results
    
    async def search_with_multiple_strategies(
        self, 
        query: str, 
        strategies: List[SearchStrategy] = None
    ) -> Dict[SearchStrategy, List[SearchResult]]:
        """
        Perform search with multiple strategies and compare results.
        
        Args:
            query: Search query
            strategies: List of strategies to try
            
        Returns:
            Dictionary mapping strategies to their results
        """
        if strategies is None:
            strategies = [SearchStrategy.STRICT, SearchStrategy.BALANCED, SearchStrategy.BROAD]
        
        results = {}
        
        for strategy in strategies:
            config = SearchConfig(strategy=strategy)
            strategy_results = await self.search(query, config)
            results[strategy] = strategy_results
            
            logger.info(f"{strategy} strategy returned {len(strategy_results)} results")
        
        return results
    
    async def get_search_suggestions(self, query: str) -> List[str]:
        """Get search suggestions based on available documents."""
        try:
            # Get a broad search to find related terms
            config = SearchConfig(strategy=SearchStrategy.BROAD, max_results=20)
            results = await self.search(query, config)
            
            # Extract keywords from results
            suggestions = set()
            for result in results:
                # Extract key phrases from content
                content_words = result.content.lower().split()
                for i, word in enumerate(content_words):
                    if len(word) > 4 and word.isalpha():
                        # Add single words
                        suggestions.add(word)
                        
                        # Add bigrams
                        if i < len(content_words) - 1:
                            bigram = f"{word} {content_words[i+1]}"
                            if len(bigram) < 30:
                                suggestions.add(bigram)
            
            # Return top suggestions
            return list(suggestions)[:10]
            
        except Exception as e:
            logger.error(f"Error getting search suggestions: {e}")
            return []


# Convenience functions for common search patterns
async def search_irs_rules(query: str, strict: bool = False) -> List[SearchResult]:
    """Search for IRS rules with appropriate configuration."""
    engine = SimilaritySearchEngine()
    
    config = SearchConfig(
        strategy=SearchStrategy.STRICT if strict else SearchStrategy.BALANCED,
        document_type_filter="irs_publication",
        max_results=5 if strict else 10,
        rerank_results=True,
        include_context=True
    )
    
    return await engine.search(query, config)


async def search_business_expenses(query: str) -> List[SearchResult]:
    """Search for business expense rules."""
    engine = SimilaritySearchEngine()
    
    config = SearchConfig(
        strategy=SearchStrategy.BALANCED,
        document_type_filter="irs_publication",
        metadata_filters={"keywords": "expense"},
        max_results=8,
        rerank_results=True
    )
    
    return await engine.search(query, config)


# Test function
async def test_similarity_search():
    """Test similarity search functionality."""
    engine = SimilaritySearchEngine()
    
    # Test different strategies
    query = "business meal deduction rules"
    results = await engine.search_with_multiple_strategies(query)
    
    for strategy, strategy_results in results.items():
        print(f"\n{strategy.upper()} Strategy ({len(strategy_results)} results):")
        for i, result in enumerate(strategy_results[:3]):
            print(f"  {i+1}. Score: {result.similarity_score:.3f}")
            print(f"     Content: {result.content[:100]}...")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_similarity_search())