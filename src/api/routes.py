from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List
import uuid
import tempfile
import os
from datetime import datetime

from src.workflows.financial_analysis import app as workflow_app
from src.config.settings import settings
from src.utils.supabase_client import supabase_client
from src.utils.document_ingestion import get_document_ingestion
from src.utils.irs_data_ingestion import get_irs_data_ingestion

app = FastAPI(
    title="Ashworth Engine v2", 
    version="1.0.0",
    description="AI-powered financial intelligence platform with Supabase backend"
)

# Response Models
class ReportSummary(BaseModel):
    """Financial analysis report summary response"""
    report_id: str = Field(..., description="Unique identifier for the report")
    status: str = Field(..., description="Current status of the report processing")
    summary: Optional[dict] = Field(None, description="Report summary metrics and data")
    warnings: Optional[List[str]] = Field(None, description="Processing warnings encountered")
    report_url: Optional[str] = Field(None, description="Signed URL for report download")
    error_message: Optional[str] = Field(None, description="Error message if processing failed")
    storage_path: Optional[str] = Field(None, description="Internal storage path for the report")

class DocumentUploadResponse(BaseModel):
    """Document upload and ingestion response"""
    document_id: str = Field(..., description="Unique identifier for the uploaded document")
    status: str = Field(..., description="Upload and ingestion status")
    filename: str = Field(..., description="Original filename of the uploaded document")
    collection_name: str = Field(..., description="RAG collection where document was stored")
    namespace: str = Field(..., description="Namespace within the collection")
    chunks_created: int = Field(..., description="Number of text chunks created from the document")
    ingestion_timestamp: str = Field(..., description="ISO timestamp when ingestion completed")
    file_size_bytes: int = Field(..., description="Size of the uploaded file in bytes")
    metadata: Optional[dict] = Field(None, description="Additional document metadata")
    error_message: Optional[str] = Field(None, description="Error message if processing failed")

class DocumentSearchResponse(BaseModel):
    """Document search results response"""
    query: str = Field(..., description="Original search query")
    results: List[dict] = Field(..., description="List of matching document chunks with scores")
    collection_name: str = Field(..., description="Collection that was searched")
    namespace: str = Field(..., description="Namespace that was searched")
    search_timestamp: str = Field(..., description="ISO timestamp when search was performed")

# Request Models
class FinancialReportRequest(BaseModel):
    """Request parameters for creating a financial analysis report"""
    client_id: str = Field(..., description="Unique identifier for the client")
    analysis_type: str = Field("financial_analysis", description="Type of analysis to perform")
    
class DocumentSearchRequest(BaseModel):
    """Request parameters for searching documents"""
    query: str = Field(..., description="Natural language search query")
    collection_name: str = Field("user_documents", description="Collection to search within")
    namespace: str = Field("default", description="Namespace to search within")
    top_k: int = Field(5, ge=1, le=50, description="Maximum number of results to return")
    score_threshold: float = Field(0.7, ge=0.0, le=1.0, description="Minimum similarity score threshold")

class TextUploadRequest(BaseModel):
    """Request parameters for uploading text content"""
    text: str = Field(..., description="Text content to upload and ingest")
    title: str = Field(..., description="Title for the text document")
    collection_name: str = Field("user_documents", description="Target collection for storage")
    namespace: str = Field("default", description="Target namespace for storage")
    client_id: Optional[str] = Field(None, description="Client identifier for document attribution")
    document_type: Optional[str] = Field(None, description="Type classification for the document")
    description: Optional[str] = Field(None, description="Optional description of the document")

class DocumentUploadRequest(BaseModel):
    """Request parameters for uploading document files"""
    collection_name: str = Field("user_documents", description="Target collection for storage")
    namespace: str = Field("default", description="Target namespace for storage")
    client_id: Optional[str] = Field(None, description="Client identifier for document attribution")
    document_type: Optional[str] = Field(None, description="Type classification for the document")
    description: Optional[str] = Field(None, description="Optional description of the document")

@app.post("/reports", response_model=ReportSummary, operation_id="createFinancialReport")
async def create_report(
    file: UploadFile = File(..., description="Financial data file (CSV, Excel, etc.)"),
    client_id: str = Form(..., description="Unique identifier for the client"),
    analysis_type: str = Form("financial_analysis", description="Type of financial analysis to perform")
):
    """Create a new financial analysis report from uploaded data"""
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
            
            # Check if upload was successful 
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

@app.get("/reports/{report_id}", operation_id="getReportStatus")
async def get_report_status(report_id: str):
    """Get detailed status and metadata for a specific financial report"""
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

@app.get("/health", operation_id="healthCheck")
async def health_check():
    """Health check endpoint with system connectivity status"""
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

# RAG Document Management Endpoints

@app.post("/rag/documents/upload", response_model=DocumentUploadResponse, operation_id="uploadDocumentFile")
async def upload_document_to_rag(
    file: UploadFile = File(..., description="Document file to upload and ingest"),
    collection_name: str = Form("user_documents", description="Target collection for document storage"),
    namespace: str = Form("default", description="Target namespace within the collection"),
    client_id: Optional[str] = Form(None, description="Client identifier for document attribution"),
    document_type: Optional[str] = Form(None, description="Type classification for the document"),
    description: Optional[str] = Form(None, description="Optional description of the document content")
):
    """Upload and ingest a document file into the RAG system"""
    try:
        # Validate file size
        file_content = await file.read()
        if len(file_content) > settings.max_upload_size:
            raise HTTPException(
                status_code=413, 
                detail=f"File too large. Maximum size: {settings.max_upload_size / (1024*1024):.1f}MB"
            )
        
        # Validate file type
        if file.filename is None:
            raise HTTPException(
                status_code=400,
                detail="Filename is required"
            )
        
        file_extension = os.path.splitext(file.filename)[1].lower()
        supported_types = settings.supported_doc_types
        
        if file_extension.lstrip('.') not in supported_types:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file_extension}. Supported types: {supported_types}"
            )
        
        # Validate collection name
        valid_collections = list(settings.rag_collection_names.values())
        if collection_name not in valid_collections:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid collection. Valid collections: {valid_collections}"
            )
        
        # Generate document ID
        document_id = str(uuid.uuid4())
        
        # Prepare metadata
        file_metadata = {
            "document_id": document_id,
            "original_filename": file.filename,
            "file_size_bytes": len(file_content),
            "upload_timestamp": datetime.utcnow().isoformat(),
            "client_id": client_id,
            "document_type": document_type,
            "description": description,
            "collection_name": collection_name,
            "namespace": namespace
        }
        
        # Create temporary file for processing
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
            tmp_file.write(file_content)
            tmp_file_path = tmp_file.name
        
        try:
            # Get document ingestion instance
            ingestion = get_document_ingestion()
            
            # Ingest the document
            doc_ids = await ingestion.ingest_file(
                file_path=tmp_file_path,
                collection_name=collection_name,
                namespace=namespace,
                additional_metadata=file_metadata
            )
            
            # Store file in Supabase Storage for audit/backup
            storage_path = None
            try:
                storage_file_path = f"rag_documents/{collection_name}/{namespace}/{document_id}_{file.filename}"
                result = supabase_client.storage.from_("documents").upload(
                    storage_file_path, file_content
                )
                storage_path = storage_file_path
            except Exception as storage_error:
                # Continue even if storage fails
                pass
            
            return DocumentUploadResponse(
                document_id=document_id,
                status="success",
                filename=file.filename or "unknown",
                collection_name=collection_name,
                namespace=namespace,
                chunks_created=len(doc_ids),
                ingestion_timestamp=datetime.utcnow().isoformat(),
                file_size_bytes=len(file_content),
                metadata={
                    **file_metadata,
                    "chunk_ids": doc_ids,
                    "storage_path": storage_path
                }
            )
            
        finally:
            # Clean up temporary file
            try:
                os.unlink(tmp_file_path)
            except:
                pass
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Document ingestion failed: {str(e)}")

@app.post("/rag/documents/upload-text", response_model=DocumentUploadResponse, operation_id="uploadTextContent")
async def upload_text_to_rag(request: TextUploadRequest):
    """Upload and ingest raw text content directly into the RAG system"""
    try:
        # Validate text length
        if len(request.text) > settings.max_upload_size:
            raise HTTPException(
                status_code=413,
                detail=f"Text too long. Maximum size: {settings.max_upload_size / (1024*1024):.1f}MB"
            )
        
        # Validate collection name
        valid_collections = list(settings.rag_collection_names.values())
        if request.collection_name not in valid_collections:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid collection. Valid collections: {valid_collections}"
            )
        
        # Generate document ID
        document_id = str(uuid.uuid4())
        
        # Prepare metadata
        text_metadata = {
            "document_id": document_id,
            "title": request.title,
            "character_count": len(request.text),
            "upload_timestamp": datetime.utcnow().isoformat(),
            "client_id": request.client_id,
            "document_type": request.document_type or "text",
            "description": request.description,
            "collection_name": request.collection_name,
            "namespace": request.namespace,
            "source": "direct_text_upload"
        }
        
        # Get document ingestion instance
        ingestion = get_document_ingestion()
        
        # Ingest the text
        doc_ids = await ingestion.ingest_text(
            text=request.text,
            collection_name=request.collection_name,
            namespace=request.namespace,
            metadata=text_metadata,
            title=request.title
        )
        
        return DocumentUploadResponse(
            document_id=document_id,
            status="success",
            filename=f"{request.title}.txt",
            collection_name=request.collection_name,
            namespace=request.namespace,
            chunks_created=len(doc_ids),
            ingestion_timestamp=datetime.utcnow().isoformat(),
            file_size_bytes=len(request.text.encode('utf-8')),
            metadata={
                **text_metadata,
                "chunk_ids": doc_ids
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Text ingestion failed: {str(e)}")

@app.post("/rag/search", response_model=DocumentSearchResponse, operation_id="searchDocuments")
async def search_rag_documents(request: DocumentSearchRequest):
    """Search documents in the RAG system using semantic similarity"""
    try:
        # Validate collection name
        valid_collections = list(settings.rag_collection_names.values())
        if request.collection_name not in valid_collections:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid collection. Valid collections: {valid_collections}"
            )
        
        # Import here to avoid circular imports
        from src.utils.vector_operations import get_vector_store
        
        # Get vector store and search
        vector_store = get_vector_store(request.collection_name)
        vector_store.initialize()
        
        results = await vector_store.similarity_search(
            query=request.query,
            k=request.top_k,
            threshold=request.score_threshold,
            namespace=request.namespace
        )
        
        # Format results
        formatted_results = []
        for content, score, metadata in results:
            formatted_results.append({
                "content": content,
                "metadata": metadata,
                "similarity_score": score
            })
        
        return DocumentSearchResponse(
            query=request.query,
            results=formatted_results,
            collection_name=request.collection_name,
            namespace=request.namespace,
            search_timestamp=datetime.utcnow().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.post("/rag/setup-irs-knowledge", operation_id="setupIrsKnowledge")
async def setup_irs_knowledge():
    """Initialize the default IRS knowledge base for comprehensive tax guidance"""
    try:
        irs_ingestion = get_irs_data_ingestion()
        doc_ids = await irs_ingestion.setup_default_irs_knowledge()
        
        return {
            "status": "success",
            "message": "IRS knowledge base setup completed",
            "documents_ingested": len(doc_ids),
            "setup_timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"IRS knowledge setup failed: {str(e)}")

@app.get("/rag/collections", operation_id="getRagCollections")
async def get_rag_collections():
    """Get available RAG collections and their configuration settings"""
    return {
        "collections": settings.rag_collection_names,
        "configuration": {
            "chunk_size": settings.rag_chunk_size,
            "chunk_overlap": settings.rag_chunk_overlap,
            "top_k_default": settings.rag_top_k,
            "score_threshold_default": settings.rag_score_threshold,
            "max_file_size_mb": settings.max_upload_size / (1024*1024),
            "supported_file_types": settings.supported_doc_types
        }
    }

@app.get("/clients/{client_id}/reports", operation_id="getClientReports")
async def get_client_reports(client_id: str, limit: int = 10):
    """Get a list of financial reports for a specific client"""
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