"""Base agent framework with personality-driven prompts and error handling."""

import logging
import traceback
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from src.workflows.state_schemas import WorkflowState, AgentStatus, update_agent_state
from src.config.settings import settings
from src.agents.personality import AgentPersonality
from src.agents.communication import AgentCommunicator
from src.agents.metrics import AgentMetrics


class BaseAgent(ABC):
    """Base class for all Ashworth Engine agents with personality and error handling."""
    
    def __init__(self, personality: AgentPersonality):
        self.personality = personality
        self.logger = self._setup_logger()
        self.memory: Dict[str, Any] = {}
        self.metrics = AgentMetrics(personality.name)
        self.communicator = AgentCommunicator()
        
    def _setup_logger(self) -> logging.Logger:
        """Set up agent-specific logger."""
        logger = logging.getLogger(f"ashworth.agents.{self.personality.name.lower().replace(' ', '_')}")
        logger.setLevel(getattr(logging, settings.log_level.upper()))
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                f'%(asctime)s - {self.personality.name} - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    @abstractmethod
    async def execute(self, state: WorkflowState) -> WorkflowState:
        """Execute the agent's main task.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated workflow state
        """
        pass
    
    @abstractmethod
    def get_agent_id(self) -> str:
        """Get the agent's unique identifier."""
        pass
    
    async def run(self, state: WorkflowState) -> WorkflowState:
        """Main execution wrapper with error handling and state management.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated workflow state
        """
        agent_id = self.get_agent_id()
        start_time = datetime.utcnow()
        
        try:
            # Update state to in progress
            state = update_agent_state(state, agent_id, AgentStatus.IN_PROGRESS)
            
            self.logger.info(f"Starting execution for {self.personality.name}")
            
            # Validate prerequisites
            self._validate_prerequisites(state)
            
            # Execute main task
            state = await self.execute(state)
            
            # Update state to completed
            state = update_agent_state(state, agent_id, AgentStatus.COMPLETED)
            
            # Log success metrics
            duration = (datetime.utcnow() - start_time).total_seconds()
            self.logger.info(f"Completed execution in {duration:.2f}s")
            self.metrics.log_success_metrics(duration, state)
            
        except Exception as e:
            # Handle error with personality-appropriate response
            error_message = self.personality.get_error_response(e, {"state": state})
            
            # Update state to failed
            state = update_agent_state(state, agent_id, AgentStatus.FAILED, error_message=error_message)
            
            # Log error details
            self.logger.error(f"Execution failed: {error_message}")
            self.logger.debug(f"Full traceback: {traceback.format_exc()}")
            
            # Add error to workflow errors
            if "errors" not in state:
                state["errors"] = []
            state["errors"].append(f"{self.personality.name}: {error_message}")
            
            # Log error metrics
            duration = (datetime.utcnow() - start_time).total_seconds()
            self.metrics.log_error_metrics(e, duration, state)
            
            # Decide whether to continue or abort workflow
            if self._should_abort_workflow(e, state):
                state["status"] = "failed"
                self.logger.critical("Aborting workflow due to critical error")
            else:
                self.logger.warning("Continuing workflow with partial data")
        
        return state
    
    def _validate_prerequisites(self, state: WorkflowState) -> None:
        """Validate that prerequisites for this agent are met.
        
        Args:
            state: Current workflow state
            
        Raises:
            ValueError: If prerequisites are not met
        """
        # Base implementation - can be overridden by specific agents
        if not state.get("workflow_id"):
            raise ValueError("Workflow ID is required")
    
    def _should_abort_workflow(self, error: Exception, state: WorkflowState) -> bool:
        """Determine if the workflow should be aborted due to this error.
        
        Args:
            error: The exception that occurred
            state: Current workflow state
            
        Returns:
            True if workflow should be aborted
        """
        # Critical errors that should abort the workflow
        critical_errors = (
            ValueError,  # Data validation errors
            FileNotFoundError,  # Missing required files
            PermissionError,  # Access issues
        )
        
        # Check if this is a critical error type
        if isinstance(error, critical_errors):
            return True
        
        # Check error count - abort if too many errors
        error_count = len(state.get("errors", []))
        if error_count >= 3:
            return True
        
        return False
    

    
    def update_memory(self, key: str, value: Any) -> None:
        """Update agent's persistent memory.
        
        Args:
            key: Memory key
            value: Value to store
        """
        self.memory[key] = value
        self.logger.debug(f"Updated memory: {key}")
    
    def get_memory(self, key: str, default: Any = None) -> Any:
        """Retrieve value from agent's memory.
        
        Args:
            key: Memory key
            default: Default value if key not found
            
        Returns:
            Stored value or default
        """
        return self.memory.get(key, default)
    
    def communicate_with_agent(self, state: WorkflowState, target_agent: str, message: str, data: Optional[Dict[str, Any]] = None) -> WorkflowState:
        """Send a message to another agent through the workflow state."""
        state = self.communicator.send_message(state, self.get_agent_id(), target_agent, message, data)
        self.logger.info(f"Sent message to {target_agent}: {message}")
        return state
    
    def get_messages_for_agent(self, state: WorkflowState, agent_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get messages addressed to this agent or a specific agent."""
        target_agent = agent_id or self.get_agent_id()
        return self.communicator.get_messages_for_agent(state, target_agent)
    
    def add_quality_metric(self, name: str, value: Union[float, int, str]) -> None:
        """Add a quality metric for this agent's execution."""
        self.metrics.add_metric(name, value)
    
    def get_quality_score(self) -> float:
        """Calculate overall quality score for this agent's execution."""
        return self.metrics.get_quality_score()


