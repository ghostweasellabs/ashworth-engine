"""CLI utilities for LLM configuration and testing."""

import asyncio
import json
import sys
import os
from typing import Optional, Dict, Any
import click
from tabulate import tabulate

from .config_manager import get_config_manager, ProviderType
from .factory import create_llm_router
from .base import ModelTier
from src.config.settings import settings


@click.group()
def llm_cli():
    """LLM configuration and testing utilities.
    
    This CLI provides comprehensive tools for managing and testing the multi-LLM
    router system with support for OpenAI, Google, and Ollama providers.
    
    Common workflows:
    
    \b
    # Check system status
    uv run python -m src.utils.llm.cli status
    
    \b
    # Test connectivity
    uv run python -m src.utils.llm.cli quick-test
    
    \b
    # Monitor performance
    uv run python -m src.utils.llm.cli monitor --watch
    
    \b
    # Configure preferences
    uv run python -m src.utils.llm.cli set-preference heavy gpt-5
    """
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
@click.option("--format", type=click.Choice(["table", "json"]), default="table", help="Output format")
@click.option("--show-sensitive", is_flag=True, help="Show sensitive values (API keys)")
def show_config(format: str, show_sensitive: bool):
    """Show current LLM configuration."""
    config_manager = get_config_manager()
    config = config_manager.get_router_config()
    
    if format == "json":
        # Mask sensitive information unless explicitly requested
        if not show_sensitive:
            masked_config = config.copy()
            for key in ["openai_api_key", "google_api_key"]:
                if masked_config.get(key):
                    masked_config[key] = "***" + masked_config[key][-4:] if len(masked_config[key]) > 4 else "***"
            config = masked_config
        
        click.echo(json.dumps(config, indent=2))
    else:
        # Table format
        click.echo(click.style("LLM Configuration", fg="cyan", bold=True))
        click.echo("=" * 50)
        
        # Provider Configuration
        click.echo(click.style("\nProvider Configuration:", fg="yellow", bold=True))
        provider_data = []
        
        ollama_host = config.get("ollama_host", "Not configured")
        provider_data.append(["Ollama Host", ollama_host])
        
        openai_key = config.get("openai_api_key")
        if openai_key:
            display_key = openai_key if show_sensitive else f"***{openai_key[-4:]}"
        else:
            display_key = "Not configured"
        provider_data.append(["OpenAI API Key", display_key])
        
        google_key = config.get("google_api_key")
        if google_key:
            display_key = google_key if show_sensitive else f"***{google_key[-4:]}"
        else:
            display_key = "Not configured"
        provider_data.append(["Google API Key", display_key])
        
        click.echo(tabulate(provider_data, headers=["Setting", "Value"], tablefmt="grid"))
        
        # Model Preferences
        click.echo(click.style("\nModel Preferences:", fg="yellow", bold=True))
        model_data = [
            ["Heavy Model Preference", config.get("heavy_model_preference", "auto")],
            ["Light Model Preference", config.get("light_model_preference", "auto")],
            ["Fallback Order", ", ".join(config.get("fallback_order", []))]
        ]
        click.echo(tabulate(model_data, headers=["Setting", "Value"], tablefmt="grid"))
        
        # Performance Settings
        click.echo(click.style("\nPerformance Settings:", fg="yellow", bold=True))
        perf_data = [
            ["Metrics Logging", "Enabled" if config.get("enable_metrics_logging") else "Disabled"],
            ["Metrics Log File", config.get("metrics_log_file", "Not set")],
            ["Performance Threshold", f"{config.get('performance_threshold_seconds', 0)}s"],
            ["Cost Threshold", f"${config.get('cost_threshold_dollars', 0)}"],
            ["Max Retries", config.get("max_retries", 0)],
            ["Provider Timeout", f"{config.get('provider_timeout_seconds', 0)}s"],
            ["Retry Delay", f"{config.get('retry_delay_seconds', 0)}s"]
        ]
        click.echo(tabulate(perf_data, headers=["Setting", "Value"], tablefmt="grid"))
        
        # GPT-5 Settings
        click.echo(click.style("\nGPT-5 Configuration:", fg="yellow", bold=True))
        gpt5_data = [
            ["Reasoning Effort", config.get("gpt5_reasoning_effort", "medium")],
            ["Include Reasoning", "Yes" if config.get("gpt5_include_reasoning") else "No"],
            ["Reasoning Depth", config.get("gpt5_reasoning_depth", "standard")]
        ]
        click.echo(tabulate(gpt5_data, headers=["Setting", "Value"], tablefmt="grid"))


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
@click.option("--format", type=click.Choice(["table", "json"]), default="table", help="Output format")
def health_check(format: str):
    """Check health of all LLM providers."""
    
    async def run_health_check():
        router = create_llm_router()
        
        if format != "json":
            click.echo("Checking provider health...")
        
        health_results = await router.health_check_all()
        
        if format == "json":
            click.echo(json.dumps(health_results, indent=2))
            return 0 if all(health_results.values()) else 1
        
        # Table format
        health_data = []
        all_healthy = True
        
        for provider, health in health_results.items():
            status = "✓ Healthy" if health else "✗ Unhealthy"
            status_color = "green" if health else "red"
            health_data.append([provider.capitalize(), click.style(status, fg=status_color)])
            
            if not health:
                all_healthy = False
        
        click.echo(tabulate(health_data, headers=["Provider", "Status"], tablefmt="grid"))
        
        return 0 if all_healthy else 1
    
    exit_code = asyncio.run(run_health_check())
    sys.exit(exit_code)


@llm_cli.command()
@click.option("--provider", type=click.Choice(["ollama", "openai", "google"]), help="Filter by provider")
@click.option("--tier", type=click.Choice(["light", "heavy"]), help="Filter by tier")
def list_models(provider: Optional[str], tier: Optional[str]):
    """List available models with their configurations."""
    config_manager = get_config_manager()
    
    models_data = []
    for model_name, model_config in config_manager.models.items():
        # Apply filters
        if provider and model_config.provider.value != provider:
            continue
        if tier and model_config.tier.value != tier:
            continue
        
        cost_display = f"${model_config.cost_per_1k_tokens:.4f}" if model_config.cost_per_1k_tokens > 0 else "Free"
        
        models_data.append([
            model_name,
            model_config.provider.value.capitalize(),
            model_config.tier.value.capitalize(),
            cost_display,
            model_config.max_tokens,
            model_config.description[:50] + "..." if len(model_config.description) > 50 else model_config.description
        ])
    
    if not models_data:
        click.echo("No models found matching the specified criteria.")
        return
    
    headers = ["Model", "Provider", "Tier", "Cost/1K Tokens", "Max Tokens", "Description"]
    click.echo(tabulate(models_data, headers=headers, tablefmt="grid"))


@llm_cli.command()
@click.argument("key")
@click.argument("value")
def set_preference(key: str, value: str):
    """Set model preference via environment variable."""
    valid_keys = {
        "heavy": "HEAVY_MODEL_PREFERENCE",
        "light": "LIGHT_MODEL_PREFERENCE",
        "provider": "LLM_PROVIDER",
        "fallback": "LLM_FALLBACK_ORDER"
    }
    
    if key not in valid_keys:
        click.echo(f"Invalid preference key. Valid keys: {', '.join(valid_keys.keys())}")
        sys.exit(1)
    
    env_var = valid_keys[key]
    
    # Validate the value
    config_manager = get_config_manager()
    
    if key in ["heavy", "light"]:
        if value != "auto" and value not in config_manager.models:
            click.echo(f"Invalid model '{value}'. Use 'llm list-models' to see available models.")
            sys.exit(1)
    elif key == "provider":
        if value not in ["ollama", "openai", "google"]:
            click.echo("Invalid provider. Valid providers: ollama, openai, google")
            sys.exit(1)
    elif key == "fallback":
        providers = [p.strip() for p in value.split(",")]
        for provider in providers:
            if provider not in ["ollama", "openai", "google"]:
                click.echo(f"Invalid provider in fallback order: {provider}")
                sys.exit(1)
    
    # Set environment variable
    os.environ[env_var] = value
    click.echo(f"Set {env_var}={value}")
    click.echo("Note: This change is temporary. To make it permanent, add it to your .env file.")


@llm_cli.command()
@click.option("--output", help="Output file for configuration")
def generate_env(output: Optional[str]):
    config_manager = get_config_manager()
    template = config_manager.get_configuration_template()
    
    if output:
        with open(output, 'w') as f:
            f.write(template)
        click.echo(f"Configuration template written to {output}")
    else:
        click.echo("# LLM Configuration Template")
        click.echo("# Copy these settings to your .env file and customize as needed")
        click.echo()
        click.echo(template)


@llm_cli.command()
@click.option("--provider", help="Test specific provider")
@click.option("--model", help="Test specific model")
@click.option("--iterations", default=1, help="Number of test iterations")
def quick_test(provider: Optional[str], model: Optional[str], iterations: int):
    """Quick test of LLM functionality."""
    
    async def run_quick_test():
        router = create_llm_router()
        
        test_prompt = "What is the capital of France? Answer in one word."
        
        success_count = 0
        total_time = 0
        total_cost = 0
        
        click.echo(f"Running {iterations} test(s)...")
        
        for i in range(iterations):
            try:
                response = await router.generate(
                    prompt=test_prompt,
                    tier=ModelTier.LIGHT,
                    provider=provider,
                    model=model,
                    temperature=0.1,
                    max_tokens=10
                )
                
                success_count += 1
                total_time += response.response_time
                if response.cost:
                    total_cost += response.cost
                
                if iterations == 1:
                    click.echo(f"✓ Success - Model: {response.model}")
                    click.echo(f"  Response: {response.content.strip()}")
                    click.echo(f"  Time: {response.response_time:.2f}s")
                    if response.cost:
                        click.echo(f"  Cost: ${response.cost:.6f}")
                
            except Exception as e:
                click.echo(f"✗ Test {i+1} failed: {e}")
        
        if iterations > 1:
            success_rate = success_count / iterations
            avg_time = total_time / success_count if success_count > 0 else 0
            
            click.echo(f"\nResults:")
            click.echo(f"  Success rate: {success_rate:.1%} ({success_count}/{iterations})")
            if success_count > 0:
                click.echo(f"  Average time: {avg_time:.2f}s")
                if total_cost > 0:
                    click.echo(f"  Total cost: ${total_cost:.6f}")
        
        return 0 if success_count > 0 else 1
    
    exit_code = asyncio.run(run_quick_test())
    sys.exit(exit_code)


@llm_cli.command()
@click.option("--watch", is_flag=True, help="Watch metrics in real-time")
@click.option("--interval", default=5, help="Update interval for watch mode (seconds)")
def monitor(watch: bool, interval: int):
    """Monitor LLM performance in real-time."""
    
    async def show_current_metrics():
        router = create_llm_router()
        summary = router.get_metrics_summary(hours=1)
        
        # Clear screen for watch mode
        if watch:
            os.system('clear' if os.name == 'posix' else 'cls')
        
        click.echo(click.style("LLM Performance Monitor", fg="cyan", bold=True))
        click.echo("=" * 50)
        
        # Provider health
        click.echo(click.style("\nProvider Health:", fg="yellow", bold=True))
        health_data = []
        for provider, health in summary["provider_health"].items():
            status = "✓ Online" if health else "✗ Offline"
            color = "green" if health else "red"
            health_data.append([provider.capitalize(), click.style(status, fg=color)])
        
        click.echo(tabulate(health_data, headers=["Provider", "Status"], tablefmt="grid"))
        
        # Recent performance
        if summary["provider_stats"]:
            click.echo(click.style("\nRecent Performance (Last Hour):", fg="yellow", bold=True))
            perf_data = []
            
            for provider, stats in summary["provider_stats"].items():
                perf_data.append([
                    provider.capitalize(),
                    stats['total_requests'],
                    f"{stats['success_rate']:.1%}",
                    f"{stats['avg_response_time']:.2f}s",
                    f"${stats['total_cost']:.4f}" if stats['total_cost'] > 0 else "Free"
                ])
            
            headers = ["Provider", "Requests", "Success Rate", "Avg Time", "Total Cost"]
            click.echo(tabulate(perf_data, headers=headers, tablefmt="grid"))
        
        # Active alerts
        if summary["active_alerts"]:
            click.echo(click.style("\nActive Alerts:", fg="red", bold=True))
            for alert in summary["active_alerts"]:
                click.echo(click.style(f"  ⚠️  {alert['message']}", fg="red"))
        
        if watch:
            click.echo(f"\nUpdating every {interval}s... (Press Ctrl+C to stop)")
    
    async def monitor_loop():
        try:
            while True:
                await show_current_metrics()
                if not watch:
                    break
                await asyncio.sleep(interval)
        except KeyboardInterrupt:
            click.echo("\nMonitoring stopped.")
    
    asyncio.run(monitor_loop())


@llm_cli.command()
@click.option("--detailed", is_flag=True, help="Show detailed information")
def status(detailed: bool):
    """Show comprehensive system status."""
    
    async def show_status():
        click.echo(click.style("Ashworth Engine LLM System Status", fg="cyan", bold=True))
        click.echo("=" * 60)
        
        # Configuration validation
        config_manager = get_config_manager()
        validation = config_manager.validate_configuration()
        
        click.echo(click.style("\n1. Configuration Status:", fg="yellow", bold=True))
        if validation["valid"]:
            click.echo(click.style("   ✓ Configuration is valid", fg="green"))
        else:
            click.echo(click.style("   ✗ Configuration has issues", fg="red"))
            for error in validation["errors"][:3]:  # Show first 3 errors
                click.echo(click.style(f"     - {error}", fg="red"))
        
        # Provider health
        click.echo(click.style("\n2. Provider Health:", fg="yellow", bold=True))
        router = create_llm_router()
        health_results = await router.health_check_all()
        
        for provider, health in health_results.items():
            status_icon = "✓" if health else "✗"
            status_color = "green" if health else "red"
            click.echo(f"   {click.style(status_icon, fg=status_color)} {provider.capitalize()}")
        
        # Current preferences
        click.echo(click.style("\n3. Current Preferences:", fg="yellow", bold=True))
        config = config_manager.get_router_config()
        click.echo(f"   Heavy Model: {config.get('heavy_model_preference', 'auto')}")
        click.echo(f"   Light Model: {config.get('light_model_preference', 'auto')}")
        click.echo(f"   Fallback Order: {' → '.join(config.get('fallback_order', []))}")
        
        # Recent activity
        click.echo(click.style("\n4. Recent Activity (Last Hour):", fg="yellow", bold=True))
        summary = router.get_metrics_summary(hours=1)
        
        if summary["provider_stats"]:
            total_requests = sum(stats['total_requests'] for stats in summary["provider_stats"].values())
            total_cost = sum(stats['total_cost'] for stats in summary["provider_stats"].values())
            
            click.echo(f"   Total Requests: {total_requests}")
            if total_cost > 0:
                click.echo(f"   Total Cost: ${total_cost:.4f}")
            
            if detailed:
                for provider, stats in summary["provider_stats"].items():
                    click.echo(f"   {provider.capitalize()}: {stats['total_requests']} requests, "
                             f"{stats['success_rate']:.1%} success rate")
        else:
            click.echo("   No recent activity")
        
        # Alerts
        if summary["active_alerts"]:
            click.echo(click.style("\n5. Active Alerts:", fg="red", bold=True))
            for alert in summary["active_alerts"]:
                click.echo(click.style(f"   ⚠️  {alert['message']}", fg="red"))
        
        # System info
        if detailed:
            click.echo(click.style("\n6. System Information:", fg="yellow", bold=True))
            click.echo(f"   Metrics Logging: {'Enabled' if config.get('enable_metrics_logging') else 'Disabled'}")
            click.echo(f"   Performance Threshold: {config.get('performance_threshold_seconds')}s")
            click.echo(f"   Cost Threshold: ${config.get('cost_threshold_dollars')}")
            click.echo(f"   Max Retries: {config.get('max_retries')}")
        
        click.echo()
    
    asyncio.run(show_status())


@llm_cli.command()
@click.argument("command", type=click.Choice(["start", "stop", "restart", "status"]))
def monitoring(command: str):
    """Control performance monitoring."""
    
    async def control_monitoring():
        router = create_llm_router()
        
        if command == "start":
            await router.start_monitoring()
            click.echo("✓ Performance monitoring started")
        elif command == "stop":
            await router.stop_monitoring()
            click.echo("✓ Performance monitoring stopped")
        elif command == "restart":
            await router.stop_monitoring()
            await router.start_monitoring()
            click.echo("✓ Performance monitoring restarted")
        elif command == "status":
            # Check if monitoring is active by looking at recent metrics
            summary = router.get_metrics_summary(hours=1)
            if summary["provider_stats"]:
                click.echo("✓ Monitoring is active (recent metrics found)")
            else:
                click.echo("? Monitoring status unclear (no recent metrics)")
    
    asyncio.run(control_monitoring())


if __name__ == "__main__":
    llm_cli()