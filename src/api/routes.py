"""FastAPI routes for workflow management and file processing."""

import os
import uuid
import asyncio
from datetime import datetime
from typing import List, Optional
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse

from src.api.models import (
    WorkflowCreateRequest,
    WorkflowCreateResponse,
    WorkflowStatusResponse,
    WorkflowResultsResponse,
    FileUploadResponse,
    ErrorResponse,
    AgentStatusResponse,
    ValidationErrorResponse,
    ValidationErrorDetail,
    WorkflowInterruptRequest,
    WorkflowInterruptResponse,
    WorkflowResumeRequest,
    WorkflowResumeResponse,
    WorkflowListResponse,
    WorkflowCancellationResponse
)
from src.workflows.state_schemas import (
    WorkflowState,
    WorkflowStatus,
    AgentStatus,
    create_initial_workflow_state
)
from src.config.settings import settings

# Create router
router = APIRouter()

# In-memory storage for workflows (replace with database in production)
workflows_store: dict[str, WorkflowState] = {}

# Supported file types and max size
SUPPORTED_FILE_TYPES = {
    "text/csv": ".csv",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
    "application/vnd.ms-excel": ".xls",
    "application/pdf": ".pdf"
}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB


def validate_file(file: UploadFile) -> None:
    """Validate uploaded file format and size."""
    # Check file size
    if hasattr(file, 'size') and file.size and file.size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File size {file.size} bytes exceeds maximum allowed size of {MAX_FILE_SIZE} bytes"
        )
    
    # Check file type by content type or extension
    valid_file = False
    
    # Check by content type first
    if file.content_type in SUPPORTED_FILE_TYPES:
        valid_file = True
    else:
        # Check by file extension if content type is not recognized
        if file.filename:
            file_ext = Path(file.filename).suffix.lower()
            if file_ext in SUPPORTED_FILE_TYPES.values():
                valid_file = True
    
    if not valid_file:
        supported_types = ", ".join(SUPPORTED_FILE_TYPES.values())
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{file.content_type}' for file '{file.filename}'. Supported types: {supported_types}"
        )


async def save_uploaded_file(file: UploadFile, upload_dir: str = "uploads") -> str:
    """Save uploaded file to disk and return the file path."""
    # Create upload directory if it doesn't exist
    Path(upload_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate unique filename
    file_extension = SUPPORTED_FILE_TYPES.get(file.content_type, "")
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    file_path = os.path.join(upload_dir, unique_filename)
    
    # Save file
    try:
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        return file_path
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save uploaded file: {str(e)}"
        )


async def execute_workflow_background(workflow_id: str):
    """Execute workflow in background (placeholder for actual implementation)."""
    # This is a placeholder - actual workflow execution will be implemented
    # when the LangGraph orchestrator is ready
    
    workflow = workflows_store.get(workflow_id)
    if not workflow:
        return
    
    try:
        # Update workflow status to running
        workflow["status"] = WorkflowStatus.RUNNING
        
        # Simulate agent execution (replace with actual LangGraph execution)
        agents = ["data_fetcher", "data_processor", "categorizer", "report_generator"]
        
        for agent_id in agents:
            # Update agent status to in progress
            if agent_id in workflow:
                workflow[agent_id]["status"] = AgentStatus.IN_PROGRESS
                workflow[agent_id]["started_at"] = datetime.utcnow()
            
            # Simulate processing time
            await asyncio.sleep(2)
            
            # Update agent status to completed
            if agent_id in workflow:
                workflow[agent_id]["status"] = AgentStatus.COMPLETED
                workflow[agent_id]["completed_at"] = datetime.utcnow()
                if workflow[agent_id].get("started_at"):
                    duration = (workflow[agent_id]["completed_at"] - workflow[agent_id]["started_at"]).total_seconds() * 1000
                    workflow[agent_id]["duration_ms"] = int(duration)
        
        # Update workflow status to completed
        workflow["status"] = WorkflowStatus.COMPLETED
        workflow["completed_at"] = datetime.utcnow()
        if workflow.get("started_at"):
            duration = (workflow["completed_at"] - workflow["started_at"]).total_seconds() * 1000
            workflow["duration_ms"] = int(duration)
            
    except Exception as e:
        # Update workflow status to failed
        workflow["status"] = WorkflowStatus.FAILED
        workflow["errors"].append(f"Workflow execution failed: {str(e)}")


@router.post("/workflows", response_model=WorkflowCreateResponse)
async def create_workflow(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(..., description="Financial documents to process (CSV, Excel, PDF)"),
    client_id: str = "default_client",  # In real implementation, extract from form data
    workflow_type: str = "financial_analysis"
):
    """
    Create a new financial analysis workflow with file uploads.
    
    This endpoint accepts financial documents and creates a new workflow for processing.
    The workflow will execute through multiple specialized agents:
    
    1. **Data Fetcher Agent** - Extracts data from uploaded files
    2. **Data Processor Agent** - Cleans and validates financial data
    3. **Categorizer Agent** - Applies tax categorization and compliance rules
    4. **Report Generator Agent** - Creates executive-grade financial reports
    
    **Supported File Types:**
    - CSV files (.csv)
    - Excel files (.xlsx, .xls)
    - PDF files (.pdf)
    
    **File Size Limits:**
    - Maximum file size: 50MB per file
    - Maximum total upload: 200MB
    
    **Example Usage:**
    ```bash
    curl -X POST "http://localhost:8000/api/v1/workflows" \\
         -H "accept: application/json" \\
         -H "Content-Type: multipart/form-data" \\
         -F "files=@financial_data.xlsx" \\
         -F "files=@receipts.pdf" \\
         -F "client_id=client_123" \\
         -F "workflow_type=financial_analysis"
    ```
    
    Args:
        files: List of financial documents to process
        client_id: Client identifier for tracking and access control
        workflow_type: Type of analysis workflow (default: financial_analysis)
        
    Returns:
        WorkflowCreateResponse with workflow ID and initial status
        
    Raises:
        HTTPException:
            - 400: No files provided or invalid request
            - 413: File size exceeds limits
            - 415: Unsupported file type
            - 422: Validation errors in file content
            - 500: Internal server error during file processing
    """
    
    # Validate files
    if not files:
        raise HTTPException(status_code=400, detail="At least one file must be uploaded")
    
    validation_errors = []
    for i, file in enumerate(files):
        try:
            validate_file(file)
        except HTTPException as e:
            validation_errors.append({
                "field": f"files[{i}]",
                "message": e.detail,
                "invalid_value": file.filename
            })
    
    if validation_errors:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "Validation failed",
                "details": validation_errors
            }
        )
    
    # Generate workflow ID
    workflow_id = str(uuid.uuid4())
    
    # Save uploaded files
    saved_files = []
    try:
        for file in files:
            file_path = await save_uploaded_file(file)
            saved_files.append(file_path)
    except HTTPException:
        # Clean up any saved files on error
        for file_path in saved_files:
            try:
                os.remove(file_path)
            except:
                pass
        raise
    
    # Create initial workflow state
    workflow_state = create_initial_workflow_state(
        workflow_id=workflow_id,
        workflow_type=workflow_type,
        input_files=saved_files,
        config={"client_id": client_id}
    )
    
    # Store workflow
    workflows_store[workflow_id] = workflow_state
    
    # Start workflow execution in background
    background_tasks.add_task(execute_workflow_background, workflow_id)
    
    return WorkflowCreateResponse(
        workflow_id=workflow_id,
        status=WorkflowStatus.PENDING,
        created_at=workflow_state["started_at"]
    )


@router.get("/workflows/{workflow_id}", response_model=WorkflowStatusResponse)
async def get_workflow_status(workflow_id: str):
    """
    Get detailed workflow status and progress information.
    
    This endpoint provides comprehensive status information about a workflow,
    including individual agent progress, timing data, and quality metrics.
    
    **Status Values:**
    - `pending`: Workflow created but not yet started
    - `running`: Workflow is currently executing
    - `completed`: Workflow finished successfully
    - `failed`: Workflow encountered an error
    - `cancelled`: Workflow was cancelled by user
    
    **Agent Status Tracking:**
    Each agent reports its individual status, timing, and any errors encountered.
    The response includes progress percentage and identifies the currently active agent.
    
    **Example Response:**
    ```json
    {
        "workflow_id": "123e4567-e89b-12d3-a456-426614174000",
        "status": "running",
        "progress_percentage": 75.0,
        "current_agent": "report_generator",
        "agents": [
            {
                "agent_id": "data_fetcher",
                "agent_name": "Dr. Marcus Thornfield",
                "status": "completed",
                "duration_ms": 2500
            }
        ]
    }
    ```
    
    Args:
        workflow_id: The unique identifier of the workflow
        
    Returns:
        WorkflowStatusResponse with detailed status information
        
    Raises:
        HTTPException: 404 if workflow not found
    """
    
    workflow = workflows_store.get(workflow_id)
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    # Calculate progress percentage
    agent_ids = ["data_fetcher", "data_processor", "categorizer", "report_generator"]
    completed_agents = sum(1 for agent_id in agent_ids 
                          if workflow.get(agent_id, {}).get("status") == AgentStatus.COMPLETED)
    progress_percentage = (completed_agents / len(agent_ids)) * 100
    
    # Find current agent
    current_agent = None
    for agent_id in agent_ids:
        if workflow.get(agent_id, {}).get("status") == AgentStatus.IN_PROGRESS:
            current_agent = agent_id
            break
    
    # Build agent status list
    agents = []
    for agent_id in agent_ids:
        agent_state = workflow.get(agent_id, {})
        if agent_state:
            agents.append(AgentStatusResponse(
                agent_id=agent_state.get("agent_id", agent_id),
                agent_name=agent_state.get("agent_name", agent_id.replace("_", " ").title()),
                status=agent_state.get("status", AgentStatus.NOT_STARTED),
                started_at=agent_state.get("started_at"),
                completed_at=agent_state.get("completed_at"),
                duration_ms=agent_state.get("duration_ms"),
                error_message=agent_state.get("error_message"),
                retry_count=agent_state.get("retry_count", 0)
            ))
    
    return WorkflowStatusResponse(
        workflow_id=workflow_id,
        workflow_type=workflow.get("workflow_type", "financial_analysis"),
        status=workflow.get("status", WorkflowStatus.PENDING),
        started_at=workflow.get("started_at", datetime.utcnow()),
        completed_at=workflow.get("completed_at"),
        duration_ms=workflow.get("duration_ms"),
        agents=agents,
        progress_percentage=progress_percentage,
        current_agent=current_agent,
        input_files=workflow.get("input_files", []),
        output_reports=workflow.get("output_reports", []),
        errors=workflow.get("errors", []),
        warnings=workflow.get("warnings", []),
        quality_score=workflow.get("quality_score"),
        confidence_score=workflow.get("confidence_score")
    )


@router.get("/workflows/{workflow_id}/results", response_model=WorkflowResultsResponse)
async def get_workflow_results(
    workflow_id: str,
    include_partial: bool = False
):
    """
    Get workflow results with support for partial results from in-progress workflows.
    
    This endpoint returns the final results of a completed workflow, or partial results
    if the workflow is still in progress and `include_partial=true` is specified.
    
    **Result Types:**
    - **Complete Results**: Available when workflow status is `completed`
    - **Partial Results**: Available when workflow is `running` and `include_partial=true`
    - **Error Results**: Available when workflow status is `failed` with error details
    
    **Partial Results Behavior:**
    When `include_partial=true`, the endpoint returns whatever results are available
    from completed agents, even if the workflow is still running. This is useful for
    real-time progress monitoring and early data access.
    
    **Example Usage:**
    ```bash
    # Get final results (only works when completed)
    curl "http://localhost:8000/api/v1/workflows/{workflow_id}/results"
    
    # Get partial results from running workflow
    curl "http://localhost:8000/api/v1/workflows/{workflow_id}/results?include_partial=true"
    ```
    
    Args:
        workflow_id: The unique identifier of the workflow
        include_partial: Whether to return partial results for in-progress workflows
        
    Returns:
        WorkflowResultsResponse with complete or partial results
        
    Raises:
        HTTPException:
            - 404: Workflow not found
            - 202: Workflow not ready (unless include_partial=true)
            - 500: Workflow failed with error details
    """
    
    workflow = workflows_store.get(workflow_id)
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    # Check workflow status and handle partial results
    status = workflow.get("status", WorkflowStatus.PENDING)
    
    if status == WorkflowStatus.PENDING:
        if not include_partial:
            raise HTTPException(
                status_code=202, 
                detail="Workflow is still pending. Use ?include_partial=true to get available data."
            )
    elif status == WorkflowStatus.RUNNING:
        if not include_partial:
            raise HTTPException(
                status_code=202, 
                detail="Workflow is still running. Use ?include_partial=true to get partial results."
            )
    elif status == WorkflowStatus.FAILED:
        # For failed workflows, still return partial results with error information
        pass
    elif status == WorkflowStatus.CANCELLED:
        # For cancelled workflows, return whatever was completed before cancellation
        pass
    
    # Extract results from workflow state (works for both complete and partial)
    analysis_state = workflow.get("analysis", {})
    report_state = workflow.get("report", {})
    file_processing_state = workflow.get("file_processing", {})
    
    # Build processing summary with available data
    processing_summary = {
        "total_files_processed": len(workflow.get("input_files", [])),
        "total_transactions": len(analysis_state.get("transactions", [])),
        "categories_identified": len(analysis_state.get("categories", {})),
        "anomalies_detected": len(analysis_state.get("anomalies", [])),
        "data_quality_score": workflow.get("quality_score"),
        "confidence_score": workflow.get("confidence_score"),
        "is_partial": status in [WorkflowStatus.PENDING, WorkflowStatus.RUNNING],
        "completed_agents": [
            agent_id for agent_id in ["data_fetcher", "data_processor", "categorizer", "report_generator"]
            if workflow.get(agent_id, {}).get("status") == AgentStatus.COMPLETED
        ]
    }
    
    # Add status-specific information
    if status == WorkflowStatus.FAILED:
        processing_summary["failure_reason"] = workflow.get("errors", ["Unknown error"])[-1] if workflow.get("errors") else "Unknown error"
    elif status == WorkflowStatus.CANCELLED:
        processing_summary["cancellation_reason"] = "Cancelled by user request"
    
    return WorkflowResultsResponse(
        workflow_id=workflow_id,
        status=status,
        transactions=analysis_state.get("transactions", []),
        financial_metrics=analysis_state.get("metrics", {}),
        report_content=report_state.get("content"),
        report_url=report_state.get("storage_path"),
        processing_summary=processing_summary,
        compliance_notes=analysis_state.get("compliance_issues", []),
        data_quality_issues=file_processing_state.get("validation_errors", []),
        started_at=workflow.get("started_at", datetime.utcnow()),
        completed_at=workflow.get("completed_at"),
        processing_time_ms=workflow.get("duration_ms")
    )


@router.delete("/workflows/{workflow_id}", response_model=WorkflowCancellationResponse)
async def cancel_workflow(workflow_id: str):
    """
    Cancel a running workflow and perform cleanup.
    
    This endpoint cancels a workflow that is currently pending or running.
    It performs the following cleanup operations:
    - Updates workflow status to cancelled
    - Stops any running agents (using LangGraph interrupts when available)
    - Cleans up uploaded files
    - Records cancellation timestamp and duration
    
    Args:
        workflow_id: The unique identifier of the workflow to cancel
        
    Returns:
        Confirmation message with workflow ID
        
    Raises:
        HTTPException: 
            - 404 if workflow not found
            - 400 if workflow cannot be cancelled (already completed/failed/cancelled)
    """
    
    workflow = workflows_store.get(workflow_id)
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    status = workflow.get("status", WorkflowStatus.PENDING)
    if status in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED, WorkflowStatus.CANCELLED]:
        raise HTTPException(
            status_code=400, 
            detail=f"Cannot cancel workflow with status: {status}"
        )
    
    # Update workflow status to cancelled
    workflow["status"] = WorkflowStatus.CANCELLED
    workflow["completed_at"] = datetime.utcnow()
    if workflow.get("started_at"):
        duration = (workflow["completed_at"] - workflow["started_at"]).total_seconds() * 1000
        workflow["duration_ms"] = int(duration)
    
    # Cancel any running agents
    agent_ids = ["data_fetcher", "data_processor", "categorizer", "report_generator"]
    cancelled_agents = []
    
    for agent_id in agent_ids:
        agent_state = workflow.get(agent_id, {})
        if agent_state.get("status") == AgentStatus.IN_PROGRESS:
            agent_state["status"] = AgentStatus.FAILED
            agent_state["error_message"] = "Cancelled by user request"
            agent_state["completed_at"] = datetime.utcnow()
            if agent_state.get("started_at"):
                duration = (agent_state["completed_at"] - agent_state["started_at"]).total_seconds() * 1000
                agent_state["duration_ms"] = int(duration)
            cancelled_agents.append(agent_id)
    
    # TODO: When LangGraph orchestrator is implemented, use proper interrupt mechanism:
    # if hasattr(workflow, 'langgraph_thread_id'):
    #     await langgraph_app.interrupt(workflow['langgraph_thread_id'])
    
    # Clean up uploaded files
    cleaned_files = []
    cleanup_errors = []
    
    for file_path in workflow.get("input_files", []):
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                cleaned_files.append(file_path)
        except Exception as e:
            cleanup_errors.append(f"Failed to clean up file {file_path}: {str(e)}")
            workflow.setdefault("warnings", []).append(f"Failed to clean up file {file_path}: {str(e)}")
    
    # Add cancellation details to workflow
    workflow.setdefault("messages", []).append({
        "timestamp": datetime.utcnow(),
        "type": "cancellation",
        "message": "Workflow cancelled by user request",
        "details": {
            "cancelled_agents": cancelled_agents,
            "cleaned_files": len(cleaned_files),
            "cleanup_errors": len(cleanup_errors)
        }
    })
    
    return WorkflowCancellationResponse(
        message="Workflow cancelled successfully", 
        workflow_id=workflow_id,
        cancelled_agents=cancelled_agents,
        cleaned_files=len(cleaned_files),
        cleanup_warnings=len(cleanup_errors)
    )


@router.post("/workflows/{workflow_id}/interrupt", response_model=WorkflowInterruptResponse)
async def interrupt_workflow(
    workflow_id: str,
    request: WorkflowInterruptRequest
):
    """
    Interrupt a running workflow at a specific point for human review.
    
    This endpoint implements LangGraph-style workflow interruption, allowing
    workflows to be paused at specific agents for human review or intervention.
    This is particularly useful for financial workflows where human oversight
    is required before critical operations.
    
    **Interrupt Points:**
    - `data_processor`: Review data quality before categorization
    - `categorizer`: Review tax categorizations before report generation
    - `report_generator`: Review analysis before final report creation
    
    **Example Usage:**
    ```bash
    # Interrupt before report generation for review
    curl -X POST "http://localhost:8000/api/v1/workflows/{workflow_id}/interrupt" \\
         -H "Content-Type: application/json" \\
         -d '{"interrupt_before": "report_generator", "message": "Review required"}'
    ```
    
    Args:
        workflow_id: The unique identifier of the workflow
        interrupt_before: Agent ID to interrupt before (optional)
        message: Reason for interruption (optional)
        
    Returns:
        Confirmation of interrupt request
        
    Raises:
        HTTPException:
            - 404: Workflow not found
            - 400: Workflow cannot be interrupted (wrong status)
    """
    
    workflow = workflows_store.get(workflow_id)
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    status = workflow.get("status", WorkflowStatus.PENDING)
    if status not in [WorkflowStatus.PENDING, WorkflowStatus.RUNNING]:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot interrupt workflow with status: {status}"
        )
    
    # Add interrupt request to workflow
    interrupt_request = {
        "timestamp": datetime.utcnow(),
        "type": "interrupt_request",
        "interrupt_before": request.interrupt_before,
        "message": request.message or "Manual interrupt requested",
        "status": "pending"
    }
    
    workflow.setdefault("interrupts", []).append(interrupt_request)
    workflow.setdefault("messages", []).append({
        "timestamp": datetime.utcnow(),
        "type": "interrupt",
        "message": f"Interrupt requested before {request.interrupt_before or 'next agent'}",
        "details": interrupt_request
    })
    
    # TODO: When LangGraph orchestrator is implemented, use proper interrupt mechanism:
    # if hasattr(workflow, 'langgraph_thread_id'):
    #     await langgraph_app.interrupt(
    #         workflow['langgraph_thread_id'], 
    #         interrupt_before=[request.interrupt_before] if request.interrupt_before else None
    #     )
    
    return WorkflowInterruptResponse(
        message="Interrupt request submitted",
        workflow_id=workflow_id,
        interrupt_before=request.interrupt_before,
        status="pending"
    )


@router.post("/workflows/{workflow_id}/resume", response_model=WorkflowResumeResponse)
async def resume_workflow(
    workflow_id: str,
    request: WorkflowResumeRequest
):
    """
    Resume an interrupted workflow after human review.
    
    This endpoint resumes a workflow that was previously interrupted for human review.
    It clears any pending interrupt requests and allows the workflow to continue
    from where it was paused.
    
    Args:
        workflow_id: The unique identifier of the workflow
        message: Reason for resumption (optional)
        
    Returns:
        Confirmation of resume request
        
    Raises:
        HTTPException:
            - 404: Workflow not found
            - 400: Workflow is not in an interruptible state
    """
    
    workflow = workflows_store.get(workflow_id)
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    # Check if workflow has pending interrupts
    interrupts = workflow.get("interrupts", [])
    pending_interrupts = [i for i in interrupts if i.get("status") == "pending"]
    
    if not pending_interrupts:
        raise HTTPException(
            status_code=400,
            detail="No pending interrupts found for this workflow"
        )
    
    # Mark interrupts as resolved
    for interrupt in pending_interrupts:
        interrupt["status"] = "resolved"
        interrupt["resolved_at"] = datetime.utcnow()
        interrupt["resolution_message"] = request.message or "Resumed by user"
    
    # Add resume message
    workflow.setdefault("messages", []).append({
        "timestamp": datetime.utcnow(),
        "type": "resume",
        "message": request.message or "Workflow resumed after interrupt",
        "resolved_interrupts": len(pending_interrupts)
    })
    
    # TODO: When LangGraph orchestrator is implemented, use proper resume mechanism:
    # if hasattr(workflow, 'langgraph_thread_id'):
    #     await langgraph_app.resume(workflow['langgraph_thread_id'])
    
    return WorkflowResumeResponse(
        message="Workflow resumed successfully",
        workflow_id=workflow_id,
        resolved_interrupts=len(pending_interrupts)
    )


@router.get("/workflows", response_model=WorkflowListResponse)
async def list_workflows(
    client_id: Optional[str] = None,
    status: Optional[WorkflowStatus] = None,
    limit: int = 50,
    offset: int = 0
):
    """
    List workflows with optional filtering and pagination.
    
    This endpoint provides a paginated list of workflows with optional filtering
    by client ID and status. Useful for workflow management dashboards and
    monitoring systems.
    
    **Example Usage:**
    ```bash
    # List all workflows
    curl "http://localhost:8000/api/v1/workflows"
    
    # List running workflows for specific client
    curl "http://localhost:8000/api/v1/workflows?client_id=client_123&status=running"
    
    # Paginated results
    curl "http://localhost:8000/api/v1/workflows?limit=10&offset=20"
    ```
    
    Args:
        client_id: Filter by client identifier (optional)
        status: Filter by workflow status (optional)
        limit: Maximum number of results to return (default: 50, max: 100)
        offset: Number of results to skip for pagination (default: 0)
        
    Returns:
        Paginated list of workflows with metadata
    """
    
    # Validate pagination parameters
    if limit > 100:
        limit = 100
    if limit < 1:
        limit = 1
    if offset < 0:
        offset = 0
    
    # Filter workflows
    filtered_workflows = []
    for workflow_id, workflow in workflows_store.items():
        # Apply filters
        if client_id and workflow.get("config", {}).get("client_id") != client_id:
            continue
        if status and workflow.get("status") != status:
            continue
        
        # Calculate progress for running workflows
        progress_percentage = 0.0
        if workflow.get("status") == WorkflowStatus.RUNNING:
            agent_ids = ["data_fetcher", "data_processor", "categorizer", "report_generator"]
            completed_agents = sum(1 for agent_id in agent_ids 
                                  if workflow.get(agent_id, {}).get("status") == AgentStatus.COMPLETED)
            progress_percentage = (completed_agents / len(agent_ids)) * 100
        elif workflow.get("status") == WorkflowStatus.COMPLETED:
            progress_percentage = 100.0
        
        # Add basic workflow info
        filtered_workflows.append({
            "workflow_id": workflow_id,
            "workflow_type": workflow.get("workflow_type", "financial_analysis"),
            "status": workflow.get("status", WorkflowStatus.PENDING),
            "started_at": workflow.get("started_at"),
            "completed_at": workflow.get("completed_at"),
            "duration_ms": workflow.get("duration_ms"),
            "client_id": workflow.get("config", {}).get("client_id"),
            "input_files_count": len(workflow.get("input_files", [])),
            "output_reports_count": len(workflow.get("output_reports", [])),
            "progress_percentage": progress_percentage,
            "has_errors": len(workflow.get("errors", [])) > 0,
            "has_warnings": len(workflow.get("warnings", [])) > 0,
            "has_interrupts": len([i for i in workflow.get("interrupts", []) if i.get("status") == "pending"]) > 0
        })
    
    # Sort by started_at descending (most recent first)
    filtered_workflows.sort(key=lambda w: w.get("started_at") or datetime.min, reverse=True)
    
    # Apply pagination
    total = len(filtered_workflows)
    paginated_workflows = filtered_workflows[offset:offset + limit]
    
    return WorkflowListResponse(
        workflows=paginated_workflows,
        total=total,
        limit=limit,
        offset=offset,
        has_more=offset + limit < total,
        filters_applied={
            "client_id": client_id,
            "status": status
        }
    )