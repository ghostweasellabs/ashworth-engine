"""Mock agent for testing the base framework."""

from datetime import datetime

from src.agents.base import BaseAgent
from src.agents.personality import AgentPersonality
from src.workflows.state_schemas import WorkflowState


class MockAgent(BaseAgent):
    """Mock agent for testing the base framework."""
    
    def __init__(self):
        personality = AgentPersonality(
            name="Test Agent",
            title="Testing Specialist",
            background="Specialized in framework validation and testing",
            personality_traits=["methodical", "thorough", "reliable"],
            communication_style="clear and concise",
            expertise_areas=["testing", "validation", "quality assurance"],
            system_prompt="You are a testing specialist focused on validating system functionality.",
            task_prompt_template="Execute test task: {task_description}",
            error_handling_style="professional"
        )
        super().__init__(personality)
    
    def get_agent_id(self) -> str:
        return "mock_agent"
    
    async def execute(self, state: WorkflowState) -> WorkflowState:
        """Mock execution for testing."""
        self.logger.info("Executing mock task")
        
        # Add some test metrics
        self.add_quality_metric("test_score", 0.95)
        self.add_quality_metric("validation_passed", 1.0)
        
        # Update memory
        self.update_memory("last_execution", datetime.utcnow().isoformat())
        
        # Send a test message
        state = self.communicate_with_agent(state, "test_receiver", "Mock execution completed")
        
        return state