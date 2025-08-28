"""Example demonstrating LLM configuration and monitoring features."""

import asyncio
import os
from src.utils.llm import (
    create_llm_router, 
    ModelTier, 
    get_config_manager,
    get_metrics_logger
)


async def main():
    """Demonstrate LLM configuration and monitoring."""
    
    print("=== LLM Configuration and Monitoring Example ===\n")
    
    # 1. Show configuration validation
    print("1. Configuration Validation:")
    config_manager = get_config_manager()
    validation = config_manager.validate_configuration()
    
    if validation["valid"]:
        print("✓ Configuration is valid")
    else:
        print("✗ Configuration has issues:")
        for error in validation["errors"]:
            print(f"  - {error}")
    
    print(f"Available providers: {list(validation['provider_status'].keys())}")
    
    # 2. Create router with enhanced configuration
    print("\n2. Creating Enhanced Router:")
    router = create_llm_router()
    
    print(f"Fallback order: {router.fallback_order}")
    print(f"Heavy model preference: {router.heavy_model_preference}")
    print(f"Light model preference: {router.light_model_preference}")
    print(f"Max retries: {router.max_retries}")
    
    # 3. Test provider health
    print("\n3. Provider Health Check:")
    health_results = await router.health_check_all()
    
    for provider, health in health_results.items():
        status = "✓ Healthy" if health else "✗ Unhealthy"
        print(f"  {provider}: {status}")
    
    # 4. Test generation with metrics logging
    print("\n4. Testing Generation with Metrics:")
    
    test_prompts = [
        ("Light task", "What is 2+2? Just the number.", ModelTier.LIGHT),
        ("Heavy task", "Explain quantum computing in simple terms.", ModelTier.HEAVY)
    ]
    
    for task_name, prompt, tier in test_prompts:
        try:
            print(f"\n  Testing {task_name} ({tier.value} tier):")
            response = await router.generate(
                prompt=prompt,
                tier=tier,
                temperature=0.1,
                max_tokens=100
            )
            
            print(f"    ✓ Success - Model: {response.model}")
            print(f"    Response time: {response.response_time:.2f}s")
            if response.cost:
                print(f"    Cost: ${response.cost:.6f}")
            print(f"    Response: {response.content[:100]}...")
            
        except Exception as e:
            print(f"    ✗ Failed: {e}")
    
    # 5. Show metrics summary
    print("\n5. Metrics Summary:")
    summary = router.get_metrics_summary(hours=1)
    
    if summary["provider_stats"]:
        for provider, stats in summary["provider_stats"].items():
            print(f"\n  {provider.upper()}:")
            print(f"    Requests: {stats['total_requests']}")
            print(f"    Success rate: {stats['success_rate']:.1%}")
            print(f"    Avg response time: {stats['avg_response_time']:.2f}s")
            if stats['total_cost'] > 0:
                print(f"    Total cost: ${stats['total_cost']:.6f}")
    else:
        print("  No metrics available yet")
    
    # 6. Show active alerts
    if summary["active_alerts"]:
        print("\n6. Active Performance Alerts:")
        for alert in summary["active_alerts"]:
            print(f"  ⚠️  {alert['message']}")
    else:
        print("\n6. No active performance alerts")
    
    # 7. Demonstrate configuration switching via environment
    print("\n7. Configuration Switching Example:")
    print("You can switch providers and models using environment variables:")
    print("  export LLM_PROVIDER=openai")
    print("  export HEAVY_MODEL_PREFERENCE=gpt-5")
    print("  export LIGHT_MODEL_PREFERENCE=gpt-4.1-mini")
    print("  export LLM_FALLBACK_ORDER=openai,google,ollama")
    
    current_provider = os.getenv("LLM_PROVIDER", "ollama")
    print(f"\nCurrent primary provider: {current_provider}")
    
    # 8. Export metrics example
    print("\n8. Metrics Export:")
    try:
        router.export_metrics("logs/example_metrics.json", hours=1, format="json")
        print("  ✓ Metrics exported to logs/example_metrics.json")
    except Exception as e:
        print(f"  ✗ Export failed: {e}")
    
    print("\n=== Example Complete ===")
    print("\nTo run the CLI tools, use:")
    print("  uv run python -m src.utils.llm.cli --help")


if __name__ == "__main__":
    asyncio.run(main())