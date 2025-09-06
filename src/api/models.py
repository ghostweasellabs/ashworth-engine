"""Pydantic models for FastAPI request/response validation."""

from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator
from enum import Enum

from src.workflows.state_schemas import WorkflowStatus, AgentStatus


class FileUploadResponse(BaseModel):
    """Response model for file upload."""
    filename: str
    size: int
    content_type: str
    upload_path: str


class WorkflowCreateRequest(BaseModel):
    """Request model for creating a new workflow."""
    client_id: str = Field(..., description="Client identifier")
    workflow_type: str = Field(default="financial_analysis", description="Type of workflow to execute")
    config: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Workflow configuration")
    
    @validator('client_id')
    def validate_client_id(cls, v):
        if not v or not v.strip():
            raise ValueError("client_id cannot be empty")
        return v.strip()


class WorkflowCreateResponse(BaseModel):
    """Response model for workflow creation."""
    workflow_id: str
    status: WorkflowStatus
    created_at: datetime
    message: str = "Workflow created successfully"
    files_processed: Optional[int] = None


class AgentStatusResponse(BaseModel):
    """Response model for individual agent status."""
    agent_id: str
    agent_name: str
    status: AgentStatus
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_ms: Optional[int] = None
    error_message: Optional[str] = None
    retry_count: int = 0


class WorkflowStatusResponse(BaseModel):
    """Response model for workflow status."""
    workflow_id: str
    workflow_type: str
    status: WorkflowStatus
    started_at: datetime
    completed_at: Optional[datetime] = None
    duration_ms: Optional[int] = None
    
    # Agent statuses
    agents: List[AgentStatusResponse]
    
    # Progress information
    progress_percentage: float = Field(ge=0.0, le=100.0)
    current_agent: Optional[str] = None
    
    # File information
    input_files: List[str]
    output_reports: List[str]
    
    # Error information
    errors: List[str]
    warnings: List[str]
    
    # Quality metrics
    quality_score: Optional[float] = None
    confidence_score: Optional[float] = None


class WorkflowResultsResponse(BaseModel):
    """Response model for workflow results."""
    workflow_id: str
    status: WorkflowStatus
    
    # Results data
    transactions: List[Dict[str, Any]] = Field(default_factory=list)
    financial_metrics: Optional[Dict[str, Any]] = None
    report_content: Optional[str] = None
    report_url: Optional[str] = None
    
    # Analysis metadata
    processing_summary: Dict[str, Any] = Field(default_factory=dict)
    compliance_notes: List[str] = Field(default_factory=list)
    data_quality_issues: List[str] = Field(default_factory=list)
    
    # Timing information
    started_at: datetime
    completed_at: Optional[datetime] = None
    processing_time_ms: Optional[int] = None


class ErrorResponse(BaseModel):
    """Standard error response model."""
    error: str
    detail: Optional[str] = None
    error_code: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class HealthCheckResponse(BaseModel):
    """Health check response model."""
    status: str
    timestamp: datetime
    services: Dict[str, str]
    version: str = "0.1.0"


class ValidationErrorDetail(BaseModel):
    """Validation error detail."""
    field: str
    message: str
    invalid_value: Any


class ValidationErrorResponse(BaseModel):
    """Validation error response."""
    error: str = "Validation failed"
    details: List[ValidationErrorDetail]
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class WorkflowInterruptRequest(BaseModel):
    """Request model for workflow interruption."""
    interrupt_before: Optional[str] = Field(
        None, 
        description="Agent ID to interrupt before (data_processor, categorizer, report_generator)"
    )
    message: Optional[str] = Field(
        None,
        description="Reason for interruption"
    )


class WorkflowInterruptResponse(BaseModel):
    """Response model for workflow interruption."""
    message: str
    workflow_id: str
    interrupt_before: Optional[str]
    status: str


class WorkflowResumeRequest(BaseModel):
    """Request model for workflow resumption."""
    message: Optional[str] = Field(
        None,
        description="Reason for resumption"
    )


class WorkflowResumeResponse(BaseModel):
    """Response model for workflow resumption."""
    message: str
    workflow_id: str
    resolved_interrupts: int


class WorkflowListResponse(BaseModel):
    """Response model for workflow listing."""
    workflows: List[Dict[str, Any]]
    total: int
    limit: int
    offset: int
    has_more: bool
    filters_applied: Dict[str, Any]


class WorkflowCancellationResponse(BaseModel):
    """Response model for workflow cancellation."""
    message: str
    workflow_id: str
    cancelled_agents: List[str]
    cleaned_files: int
    cleanup_warnings: int