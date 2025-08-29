"""Agent metrics collection and quality scoring."""

import logging
from datetime import datetime
from typing import Any, Dict, Union

from src.workflows.state_schemas import WorkflowState


class AgentMetrics:
    """Handles metrics collection and quality scoring for agents."""
    
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.metrics: Dict[str, Any] = {}
        self.logger = logging.getLogger(f"ashworth.metrics.{agent_name.lower().replace(' ', '_')}")
    
    def add_metric(self, name: str, value: Union[float, int, str]) -> None:
        """Add a quality metric.
        
        Args:
            name: Metric name
            value: Metric value
        """
        self.metrics[name] = value
        self.logger.debug(f"Added metric: {name} = {value}")
    
    def get_quality_score(self) -> float:
        """Calculate overall quality score.
        
        Returns:
            Quality score between 0.0 and 1.0
        """
        if not self.metrics:
            return 1.0
        
        # Simple average of numeric metrics
        numeric_metrics = [v for v in self.metrics.values() if isinstance(v, (int, float))]
        if not numeric_metrics:
            return 1.0
        
        return sum(numeric_metrics) / len(numeric_metrics)
    
    def log_success_metrics(self, duration: float, state: WorkflowState) -> None:
        """Log success metrics for monitoring.
        
        Args:
            duration: Execution duration in seconds
            state: Final workflow state
        """
        metrics = {
            "agent": self.agent_name,
            "status": "success",
            "duration_seconds": duration,
            "timestamp": datetime.utcnow().isoformat(),
            "workflow_id": state.get("workflow_id"),
        }
        
        # Add agent-specific metrics
        metrics.update(self.metrics)
        
        self.logger.info(f"Success metrics: {metrics}")
    
    def log_error_metrics(self, error: Exception, duration: float, state: WorkflowState) -> None:
        """Log error metrics for monitoring.
        
        Args:
            error: The exception that occurred
            duration: Execution duration in seconds
            state: Current workflow state
        """
        metrics = {
            "agent": self.agent_name,
            "status": "error",
            "error_type": type(error).__name__,
            "error_message": str(error),
            "duration_seconds": duration,
            "timestamp": datetime.utcnow().isoformat(),
            "workflow_id": state.get("workflow_id"),
        }
        
        self.logger.error(f"Error metrics: {metrics}")