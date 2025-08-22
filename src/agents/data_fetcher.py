from typing import Dict, Any, List
from src.workflows.state_schemas import OverallState, Transaction
from src.utils.file_processing import detect_file_type, parse_file
from src.utils.supabase_client import supabase_client
from src.utils.logging import StructuredLogger
from src.config.settings import settings
import io

logger = StructuredLogger()

def data_fetcher_agent(state: OverallState) -> Dict[str, Any]:
    """Extract financial data from various file formats"""
    trace_id = state.get("trace_id", "unknown")
    
    try:
        logger.log_agent_activity(
            "data_fetcher", "start_extraction", trace_id,
            analysis_type=state.get("analysis_type")
        )
        
        file_content = state.get("file_content")
        if not file_content:
            raise ValueError("No file content provided")
        
        # Detect file type and parse
        file_type = detect_file_type(file_content)
        transactions = parse_file(file_content, file_type)
        
        # Store raw data in Supabase for audit trail (optional)
        client_id = state.get("client_id")
        if client_id and transactions:
            try:
                # Store processing metadata
                supabase_client.table("analyses").insert({
                    "id": trace_id,
                    "client_id": client_id,
                    "analysis_type": state.get("analysis_type"),
                    "file_name": state.get("file_name", "uploaded_file"),
                    "file_size": len(file_content),
                    "status": "processing",
                    "results": {
                        "file_type": file_type,
                        "transaction_count": len(transactions)
                    },
                    "created_at": state.get("processing_start_time")
                }).execute()
            except Exception as db_error:
                logger.log_agent_activity(
                    "data_fetcher", "metadata_storage_failed", trace_id,
                    error=str(db_error)
                )
        
        logger.log_agent_activity(
            "data_fetcher", "extraction_complete", trace_id,
            transactions_found=len(transactions),
            file_type=file_type
        )
        
        return {
            "raw_extracted_data": [{
                "date": t.date,
                "description": t.description,
                "amount": float(t.amount),
                "account": t.account,
                "currency": t.currency,
                "category": t.category,
                "metadata": t.metadata
            } for t in transactions],
            "workflow_phase": "data_extraction_complete",
            "error_messages": []
        }
        
    except Exception as e:
        logger.log_agent_activity(
            "data_fetcher", "extraction_failed", trace_id,
            error=str(e)
        )
        return {
            "raw_extracted_data": [],
            "error_messages": [f"Data fetcher error: {str(e)}"],
            "workflow_phase": "data_extraction_failed"
        }