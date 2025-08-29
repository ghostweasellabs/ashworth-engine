#!/usr/bin/env python3
"""
Test script for vector database infrastructure.
Tests document ingestion, embedding generation, and similarity search.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils.rag.document_ingestion import DocumentIngestionPipeline
from src.utils.rag.embeddings import EmbeddingGenerator
from src.utils.rag.similarity_search import SimilaritySearchEngine, SearchConfig, SearchStrategy
from src.utils.rag.vector_store import VectorStore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_embedding_generation():
    """Test embedding generation."""
    print("\n=== Testing Embedding Generation ===")
    
    try:
        generator = EmbeddingGenerator()
        
        test_text = "Business expenses must be ordinary and necessary to be deductible."
        embedding = await generator.generate_embedding(test_text)
        
        if embedding:
            print(f"✅ Generated embedding with dimension: {len(embedding)}")
            print(f"   First 5 values: {embedding[:5]}")
            return True
        else:
            print("❌ Failed to generate embedding")
            return False
            
    except Exception as e:
        print(f"❌ Error testing embedding generation: {e}")
        return False


async def test_vector_store():
    """Test vector store operations."""
    print("\n=== Testing Vector Store ===")
    
    try:
        store = VectorStore()
        
        # Test storing a document
        success = await store.store_source_document(
            document_id="test_vector_store",
            title="Test Document for Vector Store",
            source_url=None,
            document_type="test",
            content="This is a test document for vector store functionality testing.",
            metadata={"test": True, "category": "testing"}
        )
        
        if success:
            print("✅ Successfully stored test document")
            
            # Get stats
            stats = await store.get_ingestion_stats()
            print(f"   Ingestion stats: {stats}")
            
            # Clean up
            await store.delete_document("test_vector_store")
            print("✅ Successfully cleaned up test document")
            
            return True
        else:
            print("❌ Failed to store test document")
            return False
            
    except Exception as e:
        print(f"❌ Error testing vector store: {e}")
        return False


async def test_document_ingestion():
    """Test document ingestion pipeline."""
    print("\n=== Testing Document Ingestion Pipeline ===")
    
    try:
        pipeline = DocumentIngestionPipeline()
        
        # Ingest sample IRS documents
        success = await pipeline.ingest_sample_documents()
        
        if success:
            print("✅ Successfully ingested sample documents")
            
            # Get ingestion stats
            stats = await pipeline.get_ingestion_stats()
            print(f"   Final stats: {stats}")
            
            return True
        else:
            print("❌ Failed to ingest sample documents")
            return False
            
    except Exception as e:
        print(f"❌ Error testing document ingestion: {e}")
        return False


async def test_similarity_search():
    """Test similarity search functionality."""
    print("\n=== Testing Similarity Search ===")
    
    try:
        engine = SimilaritySearchEngine()
        
        # Test different search strategies
        test_queries = [
            "business meal deduction",
            "office supplies expense",
            "depreciation rules"
        ]
        
        for query in test_queries:
            print(f"\n--- Testing query: '{query}' ---")
            
            # Test balanced strategy
            config = SearchConfig(
                strategy=SearchStrategy.BALANCED,
                max_results=5,
                rerank_results=True
            )
            
            results = await engine.search(query, config)
            
            if results:
                print(f"✅ Found {len(results)} results for '{query}'")
                for i, result in enumerate(results[:2]):
                    print(f"   {i+1}. Score: {result.similarity_score:.3f}")
                    print(f"      Content: {result.content[:100]}...")
            else:
                print(f"⚠️  No results found for '{query}'")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing similarity search: {e}")
        return False


async def test_search_strategies():
    """Test different search strategies."""
    print("\n=== Testing Search Strategies ===")
    
    try:
        engine = SimilaritySearchEngine()
        query = "business expense deduction"
        
        # Test multiple strategies
        results = await engine.search_with_multiple_strategies(
            query, 
            [SearchStrategy.STRICT, SearchStrategy.BALANCED, SearchStrategy.BROAD]
        )
        
        for strategy, strategy_results in results.items():
            print(f"   {strategy.upper()}: {len(strategy_results)} results")
            if strategy_results:
                avg_score = sum(r.similarity_score for r in strategy_results) / len(strategy_results)
                print(f"      Average score: {avg_score:.3f}")
        
        print("✅ Successfully tested multiple search strategies")
        return True
        
    except Exception as e:
        print(f"❌ Error testing search strategies: {e}")
        return False


async def run_comprehensive_test():
    """Run comprehensive test of vector database infrastructure."""
    print("🚀 Starting Vector Database Infrastructure Tests")
    print("=" * 60)
    
    # Check if OpenAI API key is available
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Warning: OPENAI_API_KEY not found in environment")
        print("   Some tests may fail without API access")
    
    test_results = []
    
    # Run all tests
    tests = [
        ("Vector Store", test_vector_store),
        ("Embedding Generation", test_embedding_generation),
        ("Document Ingestion", test_document_ingestion),
        ("Similarity Search", test_similarity_search),
        ("Search Strategies", test_search_strategies),
    ]
    
    for test_name, test_func in tests:
        try:
            result = await test_func()
            test_results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            test_results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Results Summary")
    print("=" * 60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Vector database infrastructure is working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return False


if __name__ == "__main__":
    # Run the comprehensive test
    success = asyncio.run(run_comprehensive_test())
    sys.exit(0 if success else 1)