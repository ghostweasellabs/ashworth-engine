"""State schemas for LangGraph workflows using TypedDict."""

from typing import Any, Dict, List, Optional, TypedDict
from decimal import Decimal
from datetime import datetime
from enum import Enum


class WorkflowStatus(str, Enum):
    """Workflow execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AgentStatus(str, Enum):
    """Individual agent execution status."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class FileProcessingState(TypedDict, total=False):
    """State for file processing operations."""
    file_path: str
    file_name: str
    file_size: int
    file_type: str
    raw_data: List[Dict[str, Any]]
    processed_data: List[Dict[str, Any]]
    validation_errors: List[str]
    processing_metadata: Dict[str, Any]
    status: AgentStatus
    error_message: Optional[str]


class AnalysisState(TypedDict, total=False):
    """State for financial analysis operations."""
    transactions: List[Dict[str, Any]]
    categories: Dict[str, str]
    metrics: Dict[str, Decimal]
    anomalies: List[Dict[str, Any]]
    tax_implications: Dict[str, Any]
    compliance_issues: List[str]
    status: AgentStatus
    error_message: Optional[str]


class ReportState(TypedDict, total=False):
    """State for report generation."""
    report_id: str
    report_type: str
    content: str
    visualizations: List[Dict[str, Any]]
    storage_path: str
    metadata: Dict[str, Any]
    status: AgentStatus
    error_message: Optional[str]


class AgentState(TypedDict, total=False):
    """Individual agent state tracking."""
    agent_id: str
    agent_name: str
    status: AgentStatus
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    duration_ms: Optional[int]
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    error_message: Optional[str]
    retry_count: int
    memory: Dict[str, Any]


class WorkflowState(TypedDict, total=False):
    """Main workflow state containing all agent states and data."""
    
    # Workflow metadata
    workflow_id: str
    workflow_type: str
    status: WorkflowStatus
    started_at: datetime
    completed_at: Optional[datetime]
    duration_ms: Optional[int]
    
    # Input/Output
    input_files: List[str]
    output_reports: List[str]
    
    # Agent states
    data_fetcher: AgentState
    data_processor: AgentState
    categorizer: AgentState
    report_generator: AgentState
    orchestrator: AgentState
    
    # Shared data between agents
    file_processing: FileProcessingState
    analysis: AnalysisState
    report: ReportState
    
    # Error handling
    errors: List[str]
    warnings: List[str]
    
    # Quality metrics
    quality_score: Optional[float]
    confidence_score: Optional[float]
    
    # Checkpoint metadata
    checkpoint_id: Optional[str]
    checkpoint_metadata: Dict[str, Any]
    
    # Configuration
    config: Dict[str, Any]
    
    # Messages between agents
    messages: List[Dict[str, Any]]


# Helper functions for state management
def create_initial_workflow_state(
    workflow_id: str,
    workflow_type: str = "financial_analysis",
    input_files: Optional[List[str]] = None,
    config: Optional[Dict[str, Any]] = None
) -> WorkflowState:
    """Create an initial workflow state.
    
    Args:
        workflow_id: Unique workflow identifier
        workflow_type: Type of workflow
        input_files: List of input file paths
        config: Workflow configuration
        
    Returns:
        Initial workflow state
    """
    return WorkflowState(
        workflow_id=workflow_id,
        workflow_type=workflow_type,
        status=WorkflowStatus.PENDING,
        started_at=datetime.utcnow(),
        input_files=input_files or [],
        output_reports=[],
        
        # Initialize agent states
        data_fetcher=AgentState(
            agent_id="data_fetcher",
            agent_name="Dr. Marcus Thornfield",
            status=AgentStatus.NOT_STARTED,
            retry_count=0,
            input_data={},
            output_data={},
            memory={}
        ),
        data_processor=AgentState(
            agent_id="data_processor", 
            agent_name="Dexter Blackwood",
            status=AgentStatus.NOT_STARTED,
            retry_count=0,
            input_data={},
            output_data={},
            memory={}
        ),
        categorizer=AgentState(
            agent_id="categorizer",
            agent_name="Clarke Pemberton",
            status=AgentStatus.NOT_STARTED,
            retry_count=0,
            input_data={},
            output_data={},
            memory={}
        ),
        report_generator=AgentState(
            agent_id="report_generator",
            agent_name="Professor Elena Castellanos",
            status=AgentStatus.NOT_STARTED,
            retry_count=0,
            input_data={},
            output_data={},
            memory={}
        ),
        orchestrator=AgentState(
            agent_id="orchestrator",
            agent_name="Dr. Victoria Ashworth",
            status=AgentStatus.NOT_STARTED,
            retry_count=0,
            input_data={},
            output_data={},
            memory={}
        ),
        
        # Initialize data states
        file_processing=FileProcessingState(
            status=AgentStatus.NOT_STARTED,
            validation_errors=[],
            processing_metadata={}
        ),
        analysis=AnalysisState(
            status=AgentStatus.NOT_STARTED,
            transactions=[],
            categories={},
            metrics={},
            anomalies=[],
            compliance_issues=[]
        ),
        report=ReportState(
            status=AgentStatus.NOT_STARTED,
            visualizations=[],
            metadata={}
        ),
        
        # Initialize collections
        errors=[],
        warnings=[],
        messages=[],
        
        # Configuration
        config=config or {},
        checkpoint_metadata={}
    )


def update_agent_state(
    state: WorkflowState,
    agent_id: str,
    status: AgentStatus,
    output_data: Optional[Dict[str, Any]] = None,
    error_message: Optional[str] = None
) -> WorkflowState:
    """Update an agent's state within the workflow.
    
    Args:
        state: Current workflow state
        agent_id: Agent identifier
        status: New agent status
        output_data: Agent output data
        error_message: Error message if failed
        
    Returns:
        Updated workflow state
    """
    # Get the agent state or create a new one if it doesn't exist
    agent_state = state.get(agent_id, {})
    
    # Initialize agent state if it's empty
    if not agent_state:
        agent_state = AgentState(
            agent_id=agent_id,
            agent_name=agent_id.replace("_", " ").title(),
            status=AgentStatus.NOT_STARTED,
            retry_count=0,
            input_data={},
            output_data={},
            memory={}
        )
    
    # Update status and timing
    now = datetime.utcnow()
    if status == AgentStatus.IN_PROGRESS and agent_state.get("status") == AgentStatus.NOT_STARTED:
        agent_state["started_at"] = now
    elif status in [AgentStatus.COMPLETED, AgentStatus.FAILED]:
        agent_state["completed_at"] = now
        if agent_state.get("started_at"):
            duration = (now - agent_state["started_at"]).total_seconds() * 1000
            agent_state["duration_ms"] = int(duration)
    
    # Update state
    agent_state["status"] = status
    if output_data:
        agent_state["output_data"] = output_data
    if error_message:
        agent_state["error_message"] = error_message
    
    # Update the workflow state
    state[agent_id] = agent_state
    
    return state