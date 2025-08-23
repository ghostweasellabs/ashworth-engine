from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List
import uuid
from datetime import datetime

from src.workflows.financial_analysis import app as workflow_app
from src.config.settings import settings
from src.utils.supabase_client import supabase_client

app = FastAPI(
    title="Ashworth Engine v2", 
    version="1.0.0",
    description="AI-powered financial intelligence platform with Supabase backend"
)

class ReportSummary(BaseModel):
    report_id: str
    status: str
    summary: Optional[dict] = None
    warnings: Optional[List[str]] = None
    report_url: Optional[str] = None
    error_message: Optional[str] = None
    storage_path: Optional[str] = None

@app.post("/reports", response_model=ReportSummary)
async def create_report(
    file: UploadFile = File(...),
    client_id: str = Form(...),
    analysis_type: str = Form("financial_analysis")
):
    """Create a new financial analysis report"""
    try:
        # Validate file size
        file_content = await file.read()
        if len(file_content) > settings.max_upload_size:
            raise HTTPException(
                status_code=413, 
                detail="File too large"
            )
        
        # Generate trace ID
        trace_id = str(uuid.uuid4())
        
        # Store uploaded file in Supabase Storage for audit
        storage_path = None
        try:
            file_path = f"{client_id}/{trace_id}/{file.filename}"
            
            # Reset file content and upload
            result = supabase_client.storage.from_("reports").upload(
                file_path, file_content
            )
            
            if hasattr(result, 'error') and not result.error:
                storage_path = file_path
                
        except Exception as storage_error:
            # Continue processing even if storage fails
            pass
        
        # Prepare initial state
        initial_state = {
            "client_id": client_id,
            "analysis_type": analysis_type,
            "file_content": file_content,
            "file_name": file.filename,
            "trace_id": trace_id,
            "processing_start_time": datetime.utcnow(),
            "workflow_phase": "initialized",
            "transactions": [],
            "error_messages": [],
            "warnings": [],
            "charts": []
        }
        
        # Execute workflow with proper configuration
        config = {
            "configurable": {
                "thread_id": trace_id,
                "checkpoint_ns": f"client_{client_id}"
            }
        }
        result = workflow_app.invoke(initial_state, config)
        
        # Process result
        if result.get("error_messages"):
            return ReportSummary(
                report_id=trace_id,
                status="error",
                error_message="; ".join(result["error_messages"]),
                storage_path=storage_path
            )
        
        # Generate download URL if report was stored
        report_url = None
        if result.get("final_report_pdf_path"):
            try:
                # Generate signed URL for report download
                signed_url = supabase_client.storage.from_(settings.storage_bucket).create_signed_url(
                    result["final_report_pdf_path"],
                    expires_in=3600  # 1 hour
                )
                report_url = signed_url.get('signedURL')
            except Exception:
                pass
        
        return ReportSummary(
            report_id=trace_id,
            status="completed",
            summary={
                "phase": result.get("workflow_phase"),
                "transactions_processed": len(result.get("transactions", [])),
                "charts_generated": len(result.get("charts", [])),
                "data_quality_score": result.get("data_quality_score", 0)
            },
            warnings=result.get("warnings", []),
            report_url=report_url,
            storage_path=storage_path
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/reports/{report_id}")
async def get_report_status(report_id: str):
    """Get report status and metadata from Supabase"""
    try:
        # Query Supabase for report status
        result = supabase_client.table("analyses").select("*").eq("id", report_id).execute()
        
        if not result.data:
            raise HTTPException(status_code=404, detail="Report not found")
        
        report_data = result.data[0]
        
        # Calculate progress based on status
        status_progress = {
            "initialized": 10,
            "data_extraction_complete": 30,
            "data_cleaning_complete": 50,
            "data_processing_complete": 70,
            "tax_categorization_complete": 85,
            "completed": 100,
            "error": 0
        }
        
        progress = status_progress.get(report_data.get("status", "unknown"), 0)
        
        return {
            "report_id": report_id,
            "status": report_data.get("status"),
            "progress": progress,
            "created_at": report_data.get("created_at"),
            "updated_at": report_data.get("updated_at"),
            "file_name": report_data.get("file_name"),
            "file_size": report_data.get("file_size"),
            "analysis_type": report_data.get("analysis_type"),
            "results": report_data.get("results", {})
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint with Supabase connectivity"""
    health_status = {
        "status": "healthy",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat(),
        "environment": settings.ae_env
    }
    
    # Test Supabase connectivity
    try:
        supabase_client.table("analyses").select("id").limit(1).execute()
        health_status["supabase"] = "connected"
    except Exception as e:
        health_status["supabase"] = f"error: {str(e)}"
        health_status["status"] = "degraded"
    
    # Test storage connectivity
    try:
        supabase_client.storage.list_buckets()
        health_status["storage"] = "connected"
    except Exception as e:
        health_status["storage"] = f"error: {str(e)}"
        health_status["status"] = "degraded"
    
    return health_status

@app.get("/clients/{client_id}/reports")
async def get_client_reports(client_id: str, limit: int = 10):
    """Get reports for a specific client"""
    try:
        result = supabase_client.table("analyses").select(
            "id, status, analysis_type, created_at, updated_at, file_name"
        ).eq("client_id", client_id).order("created_at", desc=True).limit(limit).execute()
        
        return {
            "client_id": client_id,
            "reports": result.data,
            "total_count": len(result.data)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)