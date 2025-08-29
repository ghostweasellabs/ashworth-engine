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
    ValidationErrorDetail
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
    files: List[UploadFile] = File(...),
    client_id: str = "default_client",  # In real implementation, extract from form data
    workflow_type: str = "financial_analysis"
):
    """Create a new workflow with file uploads."""
    
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
    """Get workflow status and progress."""
    
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
async def get_workflow_results(workflow_id: str):
    """Get final workflow results."""
    
    workflow = workflows_store.get(workflow_id)
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    # Check if workflow is completed
    status = workflow.get("status", WorkflowStatus.PENDING)
    if status == WorkflowStatus.PENDING:
        raise HTTPException(status_code=202, detail="Workflow is still pending")
    elif status == WorkflowStatus.RUNNING:
        raise HTTPException(status_code=202, detail="Workflow is still running")
    elif status == WorkflowStatus.FAILED:
        raise HTTPException(status_code=500, detail="Workflow execution failed")
    
    # Extract results from workflow state
    analysis_state = workflow.get("analysis", {})
    report_state = workflow.get("report", {})
    file_processing_state = workflow.get("file_processing", {})
    
    # Build processing summary
    processing_summary = {
        "total_files_processed": len(workflow.get("input_files", [])),
        "total_transactions": len(analysis_state.get("transactions", [])),
        "categories_identified": len(analysis_state.get("categories", {})),
        "anomalies_detected": len(analysis_state.get("anomalies", [])),
        "data_quality_score": workflow.get("quality_score"),
        "confidence_score": workflow.get("confidence_score")
    }
    
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


@router.delete("/workflows/{workflow_id}")
async def cancel_workflow(workflow_id: str):
    """Cancel a running workflow."""
    
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
    
    # Clean up uploaded files
    for file_path in workflow.get("input_files", []):
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            workflow.setdefault("warnings", []).append(f"Failed to clean up file {file_path}: {str(e)}")
    
    return {"message": "Workflow cancelled successfully", "workflow_id": workflow_id}


@router.get("/workflows")
async def list_workflows(
    client_id: Optional[str] = None,
    status: Optional[WorkflowStatus] = None,
    limit: int = 50,
    offset: int = 0
):
    """List workflows with optional filtering."""
    
    # Filter workflows
    filtered_workflows = []
    for workflow_id, workflow in workflows_store.items():
        # Apply filters
        if client_id and workflow.get("config", {}).get("client_id") != client_id:
            continue
        if status and workflow.get("status") != status:
            continue
        
        # Add basic workflow info
        filtered_workflows.append({
            "workflow_id": workflow_id,
            "workflow_type": workflow.get("workflow_type", "financial_analysis"),
            "status": workflow.get("status", WorkflowStatus.PENDING),
            "started_at": workflow.get("started_at"),
            "completed_at": workflow.get("completed_at"),
            "client_id": workflow.get("config", {}).get("client_id"),
            "input_files_count": len(workflow.get("input_files", [])),
            "output_reports_count": len(workflow.get("output_reports", []))
        })
    
    # Apply pagination
    total = len(filtered_workflows)
    paginated_workflows = filtered_workflows[offset:offset + limit]
    
    return {
        "workflows": paginated_workflows,
        "total": total,
        "limit": limit,
        "offset": offset,
        "has_more": offset + limit < total
    }