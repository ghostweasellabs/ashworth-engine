from typing import Dict, Any, Optional
from langchain_core.runnables import RunnableConfig
from src.workflows.state_schemas import OverallState, FinancialMetrics, Transaction
from src.utils.financial_calculations import calculate_metrics
from src.utils.supabase_client import supabase_client
from src.utils.logging import StructuredLogger
from decimal import Decimal

logger = StructuredLogger()

def data_processor_agent(state: OverallState, 
                        config: Optional[RunnableConfig] = None, 
                        *, 
                        store=None) -> Dict[str, Any]:
    """Process and analyze financial transactions"""
    trace_id = state.get("trace_id", "unknown")
    
    try:
        logger.log_agent_activity(
            "data_processor", "start_processing", trace_id
        )
        
        transactions = state.get("transactions", [])
        if not transactions:
            return {"error_messages": ["No transactions to process"]}
        
        # Convert dict transactions to Transaction objects if needed
        if transactions and isinstance(transactions[0], dict):
            transaction_objects = []
            for t_dict in transactions:
                try:
                    transaction = Transaction(
                        date=t_dict.get('date', '2024-01-01'),
                        description=t_dict.get('description', 'Unknown'),
                        amount=Decimal(str(t_dict.get('amount', 0))),
                        account=t_dict.get('account', 'Unknown'),
                        currency=t_dict.get('currency', 'USD'),
                        category=t_dict.get('category'),
                        metadata=t_dict.get('metadata', {})
                    )
                    transaction_objects.append(transaction)
                except Exception as e:
                    logger.log_agent_activity(
                        "data_processor", "transaction_conversion_error", trace_id,
                        error=str(e)
                    )
                    continue
            transactions = transaction_objects
        
        # Calculate financial metrics
        metrics = calculate_metrics(transactions)
        
        # Update analysis status in Supabase
        try:
            client_id = state.get("client_id")
            if client_id:
                # Convert metrics to dict for JSON storage
                metrics_dict = {
                    "total_revenue": float(metrics.total_revenue),
                    "total_expenses": float(metrics.total_expenses),
                    "gross_profit": float(metrics.gross_profit),
                    "gross_margin_pct": metrics.gross_margin_pct,
                    "expense_by_category": {k: float(v) for k, v in metrics.expense_by_category.items()},
                    "anomalies": metrics.anomalies,
                    "pattern_matches": metrics.pattern_matches,
                    "detected_business_types": metrics.detected_business_types
                }
                
                supabase_client.table("analyses").update({
                    "status": "data_processing_complete",
                    "results": metrics_dict
                }).eq("id", trace_id).execute()
        except Exception as db_error:
            logger.log_agent_activity(
                "data_processor", "status_update_failed", trace_id,
                error=str(db_error)
            )
        
        logger.log_agent_activity(
            "data_processor", "processing_complete", trace_id,
            metrics_calculated=True,
            total_revenue=float(metrics.total_revenue),
            total_expenses=float(metrics.total_expenses)
        )
        
        return {
            "financial_metrics": metrics,
            "workflow_phase": "data_processing_complete", 
            "error_messages": []
        }
        
    except Exception as e:
        logger.log_agent_activity(
            "data_processor", "processing_failed", trace_id,
            error=str(e)
        )
        return {
            "error_messages": [f"Data processor error: {str(e)}"],
            "workflow_phase": "data_processing_failed"
        }