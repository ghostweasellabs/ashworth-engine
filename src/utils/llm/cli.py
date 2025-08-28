"""CLI utilities for LLM configuration and testing."""

import asyncio
import json
import sys
from typing import Optional
import click

from .config_manager import get_config_manager
from .factory import create_llm_router
from .base import ModelTier


@click.group()
def llm_cli():
    """LLM configuration and testing utilities."""
    pass


@llm_cli.command()
def validate_config():
    """Validate current LLM configuration."""
    config_manager = get_config_manager()
    validation = config_manager.validate_configuration()
    
    if validation["valid"]:
        click.echo(click.style("✓ Configuration is valid", fg="green"))
    else:
        click.echo(click.style("✗ Configuration has errors", fg="red"))
    
    if validation["errors"]:
        click.echo("\nErrors:")
        for error in validation["errors"]:
            click.echo(click.style(f"  - {error}", fg="red"))
    
    if validation["warnings"]:
        click.echo("\nWarnings:")
        for warning in validation["warnings"]:
            click.echo(click.style(f"  - {warning}", fg="yellow"))
    
    click.echo("\nProvider Status:")
    for provider, status in validation["provider_status"].items():
        status_icon = "✓" if status["valid"] else "✗"
        status_color = "green" if status["valid"] else "red"
        click.echo(f"  {click.style(status_icon, fg=status_color)} {provider}")
        
        for warning in status["warnings"]:
            click.echo(click.style(f"    - {warning}", fg="yellow"))


@llm_cli.command()
def show_config():
    """Show current LLM configuration."""
    config_manager = get_config_manager()
    config = config_manager.get_router_config()
    
    # Mask sensitive information
    masked_config = config.copy()
    for key in ["openai_api_key", "google_api_key"]:
        if masked_config.get(key):
            masked_config[key] = "***" + masked_config[key][-4:] if len(masked_config[key]) > 4 else "***"
    
    click.echo(json.dumps(masked_config, indent=2))


@llm_cli.command()
def config_template():
    """Generate environment variable template."""
    config_manager = get_config_manager()
    template = config_manager.get_configuration_template()
    click.echo(template)


@llm_cli.command()
@click.option("--provider", help="Specific provider to test")
@click.option("--tier", type=click.Choice(["light", "heavy"]), help="Model tier to test")
@click.option("--prompt", default="What is 2+2?", help="Test prompt")
def test_provider(provider: Optional[str], tier: Optional[str], prompt: str):
    """Test LLM provider connectivity and performance."""
    
    async def run_test():
        router = create_llm_router()
        
        # Determine tier
        model_tier = ModelTier.LIGHT if tier == "light" else ModelTier.HEAVY if tier == "heavy" else ModelTier.LIGHT
        
        try:
            click.echo(f"Testing with prompt: '{prompt}'")
            click.echo(f"Tier: {model_tier.value}")
            if provider:
                click.echo(f"Provider: {provider}")
            
            response = await router.generate(
                prompt=prompt,
                tier=model_tier,
                provider=provider,
                temperature=0.1,
                max_tokens=100
            )
            
            click.echo(click.style("✓ Test successful", fg="green"))
            click.echo(f"Model: {response.model}")
            click.echo(f"Provider: {response.provider}")
            click.echo(f"Response time: {response.response_time:.2f}s")
            if response.cost:
                click.echo(f"Cost: ${response.cost:.6f}")
            if response.tokens_used:
                click.echo(f"Tokens: {response.tokens_used}")
            click.echo(f"Response: {response.content[:200]}...")
            
        except Exception as e:
            click.echo(click.style(f"✗ Test failed: {e}", fg="red"))
            return 1
        
        return 0
    
    exit_code = asyncio.run(run_test())
    sys.exit(exit_code)


@llm_cli.command()
@click.option("--hours", default=24, help="Hours of metrics to analyze")
def show_metrics(hours: int):
    """Show LLM performance metrics."""
    router = create_llm_router()
    summary = router.get_metrics_summary(hours)
    
    click.echo(f"Metrics Summary (Last {hours} hours)")
    click.echo("=" * 40)
    
    # Provider statistics
    if summary["provider_stats"]:
        click.echo("\nProvider Performance:")
        for provider, stats in summary["provider_stats"].items():
            click.echo(f"\n{provider.upper()}:")
            click.echo(f"  Requests: {stats['total_requests']}")
            click.echo(f"  Success Rate: {stats['success_rate']:.1%}")
            click.echo(f"  Avg Response Time: {stats['avg_response_time']:.2f}s")
            click.echo(f"  Total Cost: ${stats['total_cost']:.4f}")
            if stats['error_count'] > 0:
                click.echo(click.style(f"  Errors: {stats['error_count']}", fg="red"))
    
    # Active alerts
    if summary["active_alerts"]:
        click.echo(click.style("\nActive Alerts:", fg="red"))
        for alert in summary["active_alerts"]:
            click.echo(click.style(f"  - {alert['message']}", fg="red"))
    
    # Provider health
    click.echo("\nProvider Health:")
    for provider, health in summary["provider_health"].items():
        status_icon = "✓" if health else "✗"
        status_color = "green" if health else "red"
        click.echo(f"  {click.style(status_icon, fg=status_color)} {provider}")


@llm_cli.command()
@click.option("--iterations", default=3, help="Number of test iterations per model")
@click.option("--output", help="Output file for benchmark results")
def benchmark(iterations: int, output: Optional[str]):
    """Run comprehensive performance benchmark."""
    
    async def run_benchmark():
        router = create_llm_router()
        
        click.echo("Running LLM performance benchmark...")
        click.echo(f"Iterations per model: {iterations}")
        
        results = await router.run_performance_benchmark(iterations=iterations)
        
        # Display results
        for provider, provider_results in results.items():
            click.echo(f"\n{provider.upper()} Results:")
            
            for tier, tier_results in provider_results.items():
                if tier_results:
                    successful_results = [r for r in tier_results if r.get("success")]
                    if successful_results:
                        avg_time = sum(r["response_time"] for r in successful_results) / len(successful_results)
                        avg_cost = sum(r.get("cost", 0) for r in successful_results) / len(successful_results)
                        
                        click.echo(f"  {tier.replace('_', ' ').title()}:")
                        click.echo(f"    Success Rate: {len(successful_results)}/{len(tier_results)}")
                        click.echo(f"    Avg Response Time: {avg_time:.2f}s")
                        if avg_cost > 0:
                            click.echo(f"    Avg Cost: ${avg_cost:.6f}")
        
        # Save to file if requested
        if output:
            with open(output, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            click.echo(f"\nResults saved to {output}")
        
        return 0
    
    exit_code = asyncio.run(run_benchmark())
    sys.exit(exit_code)


@llm_cli.command()
@click.option("--hours", default=24, help="Hours of metrics to export")
@click.option("--format", type=click.Choice(["json", "csv"]), default="json", help="Export format")
@click.argument("output_file")
def export_metrics(hours: int, format: str, output_file: str):
    """Export metrics to file."""
    router = create_llm_router()
    
    try:
        router.export_metrics(output_file, hours, format)
        click.echo(f"Metrics exported to {output_file}")
    except Exception as e:
        click.echo(click.style(f"Export failed: {e}", fg="red"))
        sys.exit(1)


@llm_cli.command()
def health_check():
    """Check health of all LLM providers."""
    
    async def run_health_check():
        router = create_llm_router()
        
        click.echo("Checking provider health...")
        health_results = await router.health_check_all()
        
        all_healthy = True
        for provider, health in health_results.items():
            status_icon = "✓" if health else "✗"
            status_color = "green" if health else "red"
            click.echo(f"{click.style(status_icon, fg=status_color)} {provider}")
            
            if not health:
                all_healthy = False
        
        return 0 if all_healthy else 1
    
    exit_code = asyncio.run(run_health_check())
    sys.exit(exit_code)


if __name__ == "__main__":
    llm_cli()