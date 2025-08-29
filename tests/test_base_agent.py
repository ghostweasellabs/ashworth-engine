"""Tests for the base agent framework."""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import patch, MagicMock

from src.agents.base import BaseAgent
from src.agents.personality import AgentPersonality
from src.agents.mock_agent import MockAgent
from src.workflows.state_schemas import WorkflowState, AgentStatus, create_initial_workflow_state
from src.config.personas import get_personality, AGENT_PERSONALITIES


class TestAgentPersonality:
    """Test agent personality configuration."""
    
    def test_personality_creation(self):
        """Test creating an agent personality."""
        personality = AgentPersonality(
            name="Test Agent",
            title="Test Specialist",
            background="Testing background",
            personality_traits=["methodical", "thorough"],
            communication_style="clear",
            expertise_areas=["testing"],
            system_prompt="Test system prompt",
            task_prompt_template="Test task: {task}",
            error_handling_style="professional"
        )
        
        assert personality.name == "Test Agent"
        assert personality.title == "Test Specialist"
        assert "methodical" in personality.personality_traits
        assert "testing" in personality.expertise_areas
    
    def test_format_task_prompt(self):
        """Test task prompt formatting."""
        personality = AgentPersonality(
            name="Test Agent",
            title="Test Specialist", 
            background="Testing background",
            personality_traits=["methodical"],
            communication_style="clear",
            expertise_areas=["testing"],
            system_prompt="Test system prompt",
            task_prompt_template="Execute task: {task_name} with data: {data}",
            error_handling_style="professional"
        )
        
        context = {"task_name": "validation", "data": "sample_data"}
        formatted = personality.format_task_prompt(context)
        
        assert "Execute task: validation with data: sample_data" == formatted
    
    def test_format_task_prompt_missing_variable(self):
        """Test task prompt formatting with missing variables."""
        personality = AgentPersonality(
            name="Test Agent",
            title="Test Specialist",
            background="Testing background", 
            personality_traits=["methodical"],
            communication_style="clear",
            expertise_areas=["testing"],
            system_prompt="Test system prompt",
            task_prompt_template="Execute task: {task_name} with data: {missing_var}",
            error_handling_style="professional"
        )
        
        context = {"task_name": "validation"}
        formatted = personality.format_task_prompt(context)
        
        # Should return original template when variable is missing
        assert formatted == "Execute task: {task_name} with data: {missing_var}"
    
    def test_error_response_styles(self):
        """Test different error handling styles."""
        personality = AgentPersonality(
            name="Test Agent",
            title="Test Specialist",
            background="Testing background",
            personality_traits=["methodical"],
            communication_style="clear", 
            expertise_areas=["testing"],
            system_prompt="Test system prompt",
            task_prompt_template="Test task",
            error_handling_style="analytical"
        )
        
        error = ValueError("Test error")
        response = personality.get_error_response(error, {})
        
        assert "Analysis indicates a processing anomaly" in response
        assert "Test error" in response


class TestBaseAgent:
    """Test base agent functionality."""
    
    @pytest.fixture
    def mock_agent(self):
        """Create a mock agent for testing."""
        return MockAgent()
    
    @pytest.fixture
    def sample_state(self):
        """Create a sample workflow state."""
        return create_initial_workflow_state(
            workflow_id="test-workflow-123",
            workflow_type="test",
            input_files=["test.csv"],
            config={"test": True}
        )
    
    def test_agent_initialization(self, mock_agent):
        """Test agent initialization."""
        assert mock_agent.personality.name == "Test Agent"
        assert mock_agent.logger is not None
        assert isinstance(mock_agent.memory, dict)
        assert mock_agent.metrics is not None
    
    def test_get_agent_id(self, mock_agent):
        """Test agent ID retrieval."""
        assert mock_agent.get_agent_id() == "mock_agent"
    
    @pytest.mark.asyncio
    async def test_successful_execution(self, mock_agent, sample_state):
        """Test successful agent execution."""
        result_state = await mock_agent.run(sample_state)
        
        # Check that agent state was updated
        agent_state = result_state.get("mock_agent", {})
        assert agent_state.get("status") == AgentStatus.COMPLETED
        assert agent_state.get("started_at") is not None
        assert agent_state.get("completed_at") is not None
        
        # Check that metrics were added
        assert mock_agent.metrics.metrics.get("test_score") == 0.95
        assert mock_agent.metrics.metrics.get("validation_passed") == 1.0
        
        # Check that memory was updated
        assert mock_agent.get_memory("last_execution") is not None
        
        # Check that message was sent
        messages = result_state.get("messages", [])
        assert len(messages) == 1
        assert messages[0]["from"] == "mock_agent"
        assert messages[0]["to"] == "test_receiver"
    
    @pytest.mark.asyncio
    async def test_execution_with_error(self, sample_state):
        """Test agent execution with error."""
        
        class FailingAgent(BaseAgent):
            def __init__(self):
                personality = AgentPersonality(
                    name="Failing Agent",
                    title="Error Generator",
                    background="Generates errors for testing",
                    personality_traits=["error-prone"],
                    communication_style="problematic",
                    expertise_areas=["failing"],
                    system_prompt="I always fail",
                    task_prompt_template="Fail at: {task}",
                    error_handling_style="professional"
                )
                super().__init__(personality)
            
            def get_agent_id(self) -> str:
                return "failing_agent"
            
            async def execute(self, state: WorkflowState) -> WorkflowState:
                raise ValueError("Intentional test failure")
        
        failing_agent = FailingAgent()
        result_state = await failing_agent.run(sample_state)
        
        # Check that agent state shows failure
        agent_state = result_state.get("failing_agent", {})
        assert agent_state.get("status") == AgentStatus.FAILED
        assert "Intentional test failure" in agent_state.get("error_message", "")
        
        # Check that error was added to workflow errors
        errors = result_state.get("errors", [])
        assert len(errors) > 0
        assert any("Failing Agent" in error for error in errors)
    
    def test_memory_operations(self, mock_agent):
        """Test agent memory operations."""
        # Test setting memory
        mock_agent.update_memory("test_key", "test_value")
        assert mock_agent.get_memory("test_key") == "test_value"
        
        # Test getting non-existent key
        assert mock_agent.get_memory("missing_key") is None
        assert mock_agent.get_memory("missing_key", "default") == "default"
    
    def test_quality_metrics(self, mock_agent):
        """Test quality metric operations."""
        # Add metrics
        mock_agent.add_quality_metric("accuracy", 0.95)
        mock_agent.add_quality_metric("completeness", 0.88)
        mock_agent.add_quality_metric("status", "good")
        
        # Check metrics were added
        assert mock_agent.metrics.metrics["accuracy"] == 0.95
        assert mock_agent.metrics.metrics["completeness"] == 0.88
        assert mock_agent.metrics.metrics["status"] == "good"
        
        # Test quality score calculation
        quality_score = mock_agent.get_quality_score()
        expected_score = (0.95 + 0.88) / 2  # Average of numeric metrics
        assert abs(quality_score - expected_score) < 0.001
    
    def test_agent_communication(self, mock_agent, sample_state):
        """Test agent-to-agent communication."""
        # Send a message
        updated_state = mock_agent.communicate_with_agent(
            sample_state, 
            "target_agent", 
            "Test message",
            {"key": "value"}
        )
        
        # Check message was added
        messages = updated_state.get("messages", [])
        assert len(messages) == 1
        
        message = messages[0]
        assert message["from"] == "mock_agent"
        assert message["to"] == "target_agent"
        assert message["message"] == "Test message"
        assert message["data"]["key"] == "value"
        assert "timestamp" in message
    
    def test_get_messages_for_agent(self, mock_agent, sample_state):
        """Test retrieving messages for an agent."""
        # Add some messages
        sample_state["messages"] = [
            {"from": "agent1", "to": "mock_agent", "message": "Message 1"},
            {"from": "agent2", "to": "other_agent", "message": "Message 2"},
            {"from": "agent3", "to": "mock_agent", "message": "Message 3"},
        ]
        
        # Get messages for mock agent
        messages = mock_agent.get_messages_for_agent(sample_state)
        assert len(messages) == 2
        assert messages[0]["message"] == "Message 1"
        assert messages[1]["message"] == "Message 3"
        
        # Get messages for other agent
        other_messages = mock_agent.get_messages_for_agent(sample_state, "other_agent")
        assert len(other_messages) == 1
        assert other_messages[0]["message"] == "Message 2"
    
    def test_should_abort_workflow(self, mock_agent, sample_state):
        """Test workflow abort decision logic."""
        # Test with non-critical error
        non_critical_error = RuntimeError("Non-critical error")
        assert not mock_agent._should_abort_workflow(non_critical_error, sample_state)
        
        # Test with critical error
        critical_error = ValueError("Critical validation error")
        assert mock_agent._should_abort_workflow(critical_error, sample_state)
        
        # Test with too many errors
        sample_state["errors"] = ["Error 1", "Error 2", "Error 3"]
        assert mock_agent._should_abort_workflow(non_critical_error, sample_state)


class TestPersonaConfigurations:
    """Test agent persona configurations."""
    
    def test_all_personalities_exist(self):
        """Test that all required personalities are defined."""
        required_agents = [
            "data_fetcher",
            "data_processor", 
            "categorizer",
            "report_generator",
            "orchestrator"
        ]
        
        for agent_type in required_agents:
            personality = get_personality(agent_type)
            assert personality is not None
            assert personality.name is not None
            assert personality.system_prompt is not None
            assert personality.task_prompt_template is not None
    
    def test_personality_retrieval(self):
        """Test personality retrieval by agent type."""
        data_fetcher = get_personality("data_fetcher")
        assert data_fetcher.name == "Dr. Marcus Thornfield"
        assert "Wharton PhD" in data_fetcher.background
        assert "analytical" in data_fetcher.error_handling_style
    
    def test_invalid_personality_type(self):
        """Test error handling for invalid personality type."""
        with pytest.raises(ValueError, match="Unknown agent type"):
            get_personality("invalid_agent_type")
    
    def test_personality_completeness(self):
        """Test that all personalities have required fields."""
        for agent_type, personality in AGENT_PERSONALITIES.items():
            assert personality.name, f"{agent_type} missing name"
            assert personality.title, f"{agent_type} missing title"
            assert personality.background, f"{agent_type} missing background"
            assert personality.personality_traits, f"{agent_type} missing traits"
            assert personality.communication_style, f"{agent_type} missing communication style"
            assert personality.expertise_areas, f"{agent_type} missing expertise areas"
            assert personality.system_prompt, f"{agent_type} missing system prompt"
            assert personality.task_prompt_template, f"{agent_type} missing task template"
            assert personality.error_handling_style, f"{agent_type} missing error handling style"


if __name__ == "__main__":
    pytest.main([__file__])