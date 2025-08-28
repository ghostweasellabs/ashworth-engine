"""Tests for LLM CLI functionality."""

import pytest
import tempfile
import os
from unittest.mock import Mock, patch, AsyncMock
from click.testing import CliRunner

from src.utils.llm.cli import llm_cli
from src.utils.llm.base import ModelTier, LLMResponse
from src.utils.llm.config_manager import LLMConfigManager


class TestLLMCLI:
    """Test LLM CLI commands."""
    
    def setup_method(self):
        """Set up test environment."""
        self.runner = CliRunner()
    
    def test_help_command(self):
        """Test CLI help output."""
        result = self.runner.invoke(llm_cli, ['--help'])
        assert result.exit_code == 0
        assert "LLM configuration and testing utilities" in result.output
        assert "status" in result.output
        assert "quick-test" in result.output
    
    def test_list_models_command(self):
        """Test list-models command."""
        result = self.runner.invoke(llm_cli, ['list-models'])
        assert result.exit_code == 0
        assert "gpt-5" in result.output
        assert "gemini-2.5-pro" in result.output
        assert "gpt-oss:20b" in result.output
    
    def test_list_models_with_filters(self):
        """Test list-models with provider and tier filters."""
        # Test provider filter
        result = self.runner.invoke(llm_cli, ['list-models', '--provider', 'ollama'])
        assert result.exit_code == 0
        assert "gpt-oss:20b" in result.output
        assert "gpt-5" not in result.output  # Should not show OpenAI models
        
        # Test tier filter
        result = self.runner.invoke(llm_cli, ['list-models', '--tier', 'light'])
        assert result.exit_code == 0
        assert "gemma3n:e4b" in result.output
        assert "gpt-oss:20b" not in result.output  # Should not show heavy models
    
    def test_show_config_table_format(self):
        """Test show-config command with table format."""
        result = self.runner.invoke(llm_cli, ['show-config'])
        assert result.exit_code == 0
        assert "LLM Configuration" in result.output
        assert "Provider Configuration" in result.output
        assert "Model Preferences" in result.output
        assert "Performance Settings" in result.output
    
    def test_show_config_json_format(self):
        """Test show-config command with JSON format."""
        result = self.runner.invoke(llm_cli, ['show-config', '--format', 'json'])
        assert result.exit_code == 0
        # Should be valid JSON
        import json
        config = json.loads(result.output)
        assert "ollama_host" in config
        assert "heavy_model_preference" in config
    
    def test_validate_config_command(self):
        """Test validate-config command."""
        result = self.runner.invoke(llm_cli, ['validate-config'])
        assert result.exit_code == 0
        # Should show validation results
        assert "Configuration" in result.output
        assert "Provider Status" in result.output
    
    def test_generate_env_command(self):
        """Test generate-env command."""
        result = self.runner.invoke(llm_cli, ['generate-env'])
        assert result.exit_code == 0
        assert "LLM_PROVIDER" in result.output
        assert "OLLAMA_HOST" in result.output
        assert "HEAVY_MODEL_PREFERENCE" in result.output
    
    def test_generate_env_with_output_file(self):
        """Test generate-env command with output file."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            temp_file = f.name
        
        try:
            result = self.runner.invoke(llm_cli, ['generate-env', '--output', temp_file])
            assert result.exit_code == 0
            assert f"Configuration template written to {temp_file}" in result.output
            
            # Check file contents
            with open(temp_file, 'r') as f:
                content = f.read()
                assert "LLM_PROVIDER" in content
                assert "OLLAMA_HOST" in content
        finally:
            os.unlink(temp_file)
    
    def test_set_preference_valid_values(self):
        """Test set-preference command with valid values."""
        # Test setting heavy model preference
        result = self.runner.invoke(llm_cli, ['set-preference', 'heavy', 'gpt-5'])
        assert result.exit_code == 0
        assert "HEAVY_MODEL_PREFERENCE=gpt-5" in result.output
        
        # Test setting light model preference
        result = self.runner.invoke(llm_cli, ['set-preference', 'light', 'auto'])
        assert result.exit_code == 0
        assert "LIGHT_MODEL_PREFERENCE=auto" in result.output
        
        # Test setting provider
        result = self.runner.invoke(llm_cli, ['set-preference', 'provider', 'ollama'])
        assert result.exit_code == 0
        assert "LLM_PROVIDER=ollama" in result.output
    
    def test_set_preference_invalid_values(self):
        """Test set-preference command with invalid values."""
        # Invalid model
        result = self.runner.invoke(llm_cli, ['set-preference', 'heavy', 'invalid-model'])
        assert result.exit_code == 1
        assert "Invalid model" in result.output
        
        # Invalid provider
        result = self.runner.invoke(llm_cli, ['set-preference', 'provider', 'invalid-provider'])
        assert result.exit_code == 1
        assert "Invalid provider" in result.output
        
        # Invalid key
        result = self.runner.invoke(llm_cli, ['set-preference', 'invalid-key', 'value'])
        assert result.exit_code == 1
        assert "Invalid preference key" in result.output
    
    @patch('src.utils.llm.cli.create_llm_router')
    def test_health_check_command(self, mock_create_router):
        """Test health-check command."""
        # Mock router with health check results
        mock_router = Mock()
        mock_router.health_check_all = AsyncMock(return_value={
            "ollama": True,
            "openai": False,
            "google": False
        })
        mock_create_router.return_value = mock_router
        
        result = self.runner.invoke(llm_cli, ['health-check'])
        assert result.exit_code == 1  # Should fail because not all providers are healthy
        assert "Checking provider health" in result.output
    
    @patch('src.utils.llm.cli.create_llm_router')
    def test_health_check_json_format(self, mock_create_router):
        """Test health-check command with JSON format."""
        mock_router = Mock()
        mock_router.health_check_all = AsyncMock(return_value={
            "ollama": True,
            "openai": False
        })
        mock_create_router.return_value = mock_router
        
        result = self.runner.invoke(llm_cli, ['health-check', '--format', 'json'])
        assert result.exit_code == 1
        
        # Should be valid JSON without any extra messages
        import json
        health_data = json.loads(result.output.strip())
        assert health_data["ollama"] is True
        assert health_data["openai"] is False
    
    @patch('src.utils.llm.cli.create_llm_router')
    def test_quick_test_command(self, mock_create_router):
        """Test quick-test command."""
        # Mock successful response
        mock_response = LLMResponse(
            content="Paris",
            model="gemma3n:e4b",
            provider="ollama",
            tokens_used=5,
            response_time=1.5,
            cost=0.0
        )
        
        mock_router = Mock()
        mock_router.generate = AsyncMock(return_value=mock_response)
        mock_create_router.return_value = mock_router
        
        result = self.runner.invoke(llm_cli, ['quick-test'])
        assert result.exit_code == 0
        assert "✓ Success" in result.output
        assert "Model: gemma3n:e4b" in result.output
        assert "Response: Paris" in result.output
    
    @patch('src.utils.llm.cli.create_llm_router')
    def test_quick_test_with_iterations(self, mock_create_router):
        """Test quick-test command with multiple iterations."""
        mock_response = LLMResponse(
            content="Paris",
            model="gemma3n:e4b",
            provider="ollama",
            response_time=1.0,
            cost=0.0
        )
        
        mock_router = Mock()
        mock_router.generate = AsyncMock(return_value=mock_response)
        mock_create_router.return_value = mock_router
        
        result = self.runner.invoke(llm_cli, ['quick-test', '--iterations', '3'])
        assert result.exit_code == 0
        assert "Running 3 test(s)" in result.output
        assert "Success rate: 100.0%" in result.output
    
    @patch('src.utils.llm.cli.create_llm_router')
    def test_status_command(self, mock_create_router):
        """Test status command."""
        mock_router = Mock()
        mock_router.health_check_all = AsyncMock(return_value={
            "ollama": True,
            "openai": False
        })
        mock_router.get_metrics_summary = Mock(return_value={
            "provider_stats": {
                "ollama": {
                    "total_requests": 5,
                    "success_rate": 0.8,
                    "total_cost": 0.0
                }
            },
            "active_alerts": [],
            "provider_health": {"ollama": True, "openai": False}
        })
        mock_create_router.return_value = mock_router
        
        result = self.runner.invoke(llm_cli, ['status'])
        assert result.exit_code == 0
        assert "Ashworth Engine LLM System Status" in result.output
        assert "Configuration Status" in result.output
        assert "Provider Health" in result.output
        assert "Current Preferences" in result.output
    
    @patch('src.utils.llm.cli.create_llm_router')
    def test_monitor_command(self, mock_create_router):
        """Test monitor command (single run, not watch mode)."""
        mock_router = Mock()
        mock_router.get_metrics_summary = Mock(return_value={
            "provider_stats": {
                "ollama": {
                    "total_requests": 10,
                    "success_rate": 0.9,
                    "avg_response_time": 2.5,
                    "total_cost": 0.0
                }
            },
            "active_alerts": [],
            "provider_health": {"ollama": True}
        })
        mock_create_router.return_value = mock_router
        
        result = self.runner.invoke(llm_cli, ['monitor'])
        assert result.exit_code == 0
        assert "LLM Performance Monitor" in result.output
        assert "Provider Health" in result.output
    
    @patch('src.utils.llm.cli.create_llm_router')
    def test_show_metrics_command(self, mock_create_router):
        """Test show-metrics command."""
        mock_router = Mock()
        mock_router.get_metrics_summary = Mock(return_value={
            "provider_stats": {
                "ollama": {
                    "total_requests": 15,
                    "success_rate": 0.85,
                    "avg_response_time": 2.0,
                    "total_cost": 0.0,
                    "error_count": 2
                }
            },
            "active_alerts": [
                {"message": "High error rate detected"}
            ],
            "provider_health": {"ollama": True}
        })
        mock_create_router.return_value = mock_router
        
        result = self.runner.invoke(llm_cli, ['show-metrics'])
        assert result.exit_code == 0
        assert "Metrics Summary" in result.output
        assert "Provider Performance" in result.output
        assert "OLLAMA" in result.output
        assert "Active Alerts" in result.output
    
    @patch('src.utils.llm.cli.create_llm_router')
    def test_benchmark_command(self, mock_create_router):
        """Test benchmark command."""
        mock_router = Mock()
        mock_router.run_performance_benchmark = AsyncMock(return_value={
            "ollama": {
                "light_tier": [
                    {"success": True, "response_time": 1.5, "cost": 0.0},
                    {"success": True, "response_time": 1.8, "cost": 0.0}
                ],
                "heavy_tier": [
                    {"success": True, "response_time": 3.0, "cost": 0.0}
                ]
            }
        })
        mock_create_router.return_value = mock_router
        
        result = self.runner.invoke(llm_cli, ['benchmark', '--iterations', '2'])
        assert result.exit_code == 0
        assert "Running LLM performance benchmark" in result.output
        assert "OLLAMA Results" in result.output


@pytest.mark.integration
class TestLLMCLIIntegration:
    """Integration tests for LLM CLI (requires actual services)."""
    
    def setup_method(self):
        """Set up test environment."""
        self.runner = CliRunner()
    
    def test_real_list_models(self):
        """Test list-models with real configuration."""
        result = self.runner.invoke(llm_cli, ['list-models'])
        assert result.exit_code == 0
        # Should show all configured models
        assert "Model" in result.output
        assert "Provider" in result.output
        assert "Tier" in result.output
    
    def test_real_show_config(self):
        """Test show-config with real configuration."""
        result = self.runner.invoke(llm_cli, ['show-config'])
        assert result.exit_code == 0
        assert "LLM Configuration" in result.output
    
    def test_real_validate_config(self):
        """Test validate-config with real configuration."""
        result = self.runner.invoke(llm_cli, ['validate-config'])
        assert result.exit_code == 0
        # Should complete without errors
    
    @pytest.mark.skipif(
        not os.getenv("INTEGRATION_TESTS"),
        reason="Integration tests disabled"
    )
    def test_real_health_check(self):
        """Test health-check with real providers."""
        result = self.runner.invoke(llm_cli, ['health-check'])
        # Exit code depends on provider availability
        assert "Checking provider health" in result.output
    
    @pytest.mark.skipif(
        not os.getenv("INTEGRATION_TESTS"),
        reason="Integration tests disabled"
    )
    def test_real_quick_test(self):
        """Test quick-test with real providers."""
        result = self.runner.invoke(llm_cli, ['quick-test', '--provider', 'ollama'])
        # Should work if Ollama is available
        if result.exit_code == 0:
            assert "✓ Success" in result.output
        else:
            # Provider not available, which is acceptable
            assert "failed" in result.output.lower()