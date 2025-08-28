#!/usr/bin/env python3
"""Example usage of the LLM provider abstraction layer."""

import asyncio
from src.utils.llm import get_llm_router, ModelTier


async def main():
    """Demonstrate LLM provider usage."""
    # Get the global router instance
    router = get_llm_router()
    
    # Example 1: Simple generation with automatic provider selection
    print("Example 1: Simple Generation")
    response = await router.generate(
        prompt="Explain compound interest in one sentence.",
        tier=ModelTier.LIGHT,
        temperature=0.7,
        max_tokens=50
    )
    print(f"Response: {response.content}")
    print(f"Used: {response.provider}/{response.model}")
    print()
    
    # Example 2: Heavy reasoning task
    print("Example 2: Heavy Reasoning Task")
    response = await router.generate(
        prompt="Analyze the pros and cons of implementing a progressive tax system.",
        tier=ModelTier.HEAVY,
        temperature=0.8,
        max_tokens=300
    )
    print(f"Response: {response.content[:200]}...")
    print(f"Used: {response.provider}/{response.model}")
    print(f"Tokens: {response.tokens_used}, Time: {response.response_time:.2f}s")
    print()
    
    # Example 3: Specific provider selection
    print("Example 3: Specific Provider")
    try:
        response = await router.generate(
            prompt="What is machine learning?",
            tier=ModelTier.LIGHT,
            provider="ollama",  # Force local Ollama
            temperature=0.5
        )
        print(f"Response: {response.content}")
        print(f"Used: {response.provider}/{response.model}")
    except Exception as e:
        print(f"Failed: {e}")
    print()
    
    # Example 4: GPT-5 specific features (if available)
    print("Example 4: GPT-5 Thinking Model (if available)")
    try:
        response = await router.generate(
            prompt="Solve this step by step: If a train travels 60 mph for 2.5 hours, how far does it go?",
            tier=ModelTier.HEAVY,
            provider="openai",
            model="gpt-5",
            reasoning_depth="thorough",
            temperature=0.3
        )
        print(f"Response: {response.content}")
        print(f"Used: {response.provider}/{response.model}")
    except Exception as e:
        print(f"GPT-5 not available: {e}")
    print()
    
    # Example 5: Health monitoring
    print("Example 5: Provider Health Status")
    health = await router.health_check_all()
    for provider, healthy in health.items():
        status = "✅" if healthy else "❌"
        print(f"{provider}: {status}")
    print()
    
    # Example 6: Benchmarking
    print("Example 6: Performance Benchmark")
    results = await router.benchmark_providers()
    for provider, result in results.items():
        if result["success"]:
            print(f"{provider}: {result['response_time']:.2f}s ({result['model']})")
        else:
            print(f"{provider}: Failed - {result['error']}")


if __name__ == "__main__":
    asyncio.run(main())