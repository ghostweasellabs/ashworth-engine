# Phase 2: Core Development - Modular Workflow Implementation

## Duration: 5-7 days
## Goal: Develop multi-agent workflow and core services in modular structure

### 2.1 LangGraph Workflow Foundation

**Create main workflow in `src/workflows/financial_analysis.py`:**

```python
from langgraph.graph import StateGraph, START, END
from langgraph.func import entrypoint, task
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from src.workflows.state_schemas import OverallState, InputState, OutputState
from typing import Dict, Any
import uuid
from datetime import datetime

@task
def data_fetcher_task(state: OverallState) -> Dict[str, Any]:
    """Extract and normalize financial data from input files"""
    try:
        # Placeholder implementation - will be filled in this phase
        return {
            "raw_extracted_data": [],
            "workflow_phase": "data_extraction_complete",
            "error_messages": []
        }
    except Exception as e:
        return {
            "raw_extracted_data": [],
            "error_messages": [f"Data fetcher error: {str(e)}"],
            "workflow_phase": "data_extraction_failed"
        }

@task
def data_cleaner_task(state: OverallState) -> Dict[str, Any]:
    """Clean, standardize, and format extracted data for LLM processing"""
    try:
        # Placeholder implementation - will be filled in this phase
        return {
            "transactions": [],
            "data_quality_score": 0.0,
            "cleaning_summary": {},
            "workflow_phase": "data_cleaning_complete",
            "error_messages": []
        }
    except Exception as e:
        return {
            "transactions": [],
            "error_messages": [f"Data cleaner error: {str(e)}"],
            "workflow_phase": "data_cleaning_failed"
        }

@task  
def data_processor_task(state: OverallState) -> Dict[str, Any]:
    """Process and analyze financial data"""
    try:
        # Placeholder implementation
        return {
            "financial_metrics": None,
            "workflow_phase": "data_processing_complete",
            "error_messages": []
        }
    except Exception as e:
        return {
            "error_messages": [f"Data processor error: {str(e)}"],
            "workflow_phase": "data_processing_failed"
        }

@task
def tax_categorizer_task(state: OverallState) -> Dict[str, Any]:
    """Categorize transactions for tax compliance"""
    try:
        # Placeholder implementation
        return {
            "tax_summary": None,
            "workflow_phase": "tax_categorization_complete",
            "error_messages": []
        }
    except Exception as e:
        return {
            "error_messages": [f"Tax categorizer error: {str(e)}"],
            "workflow_phase": "tax_categorization_failed"
        }

@task
def report_generator_task(state: OverallState) -> Dict[str, Any]:
    """Generate consulting-grade narrative report"""
    try:
        # Placeholder implementation
        return {
            "final_report_md": "# Sample Report\n\nGenerated successfully.",
            "final_report_pdf_path": None,
            "charts": [],
            "workflow_phase": "report_generation_complete",
            "error_messages": []
        }
    except Exception as e:
        return {
            "error_messages": [f"Report generator error: {str(e)}"],
            "workflow_phase": "report_generation_failed"
        }

# Build StateGraph with proper configuration
graph_builder = StateGraph(
    OverallState,
    input_schema=InputState,
    output_schema=OutputState
)

# Add nodes with error handling
graph_builder.add_node("data_fetcher", data_fetcher_task)
graph_builder.add_node("data_cleaner", data_cleaner_task)
graph_builder.add_node("data_processor", data_processor_task)
graph_builder.add_node("tax_categorizer", tax_categorizer_task)
graph_builder.add_node("report_generator", report_generator_task)

# Define workflow edges based on analysis_type
def route_workflow(state: OverallState) -> str:
    """Route based on analysis_type parameter"""
    analysis_type = state.get("analysis_type", "financial_analysis")
    
    if analysis_type == "data_collection":
        return END  # Stop after data fetcher
    elif analysis_type == "data_cleaning":
        return "data_cleaner"  # Stop after data cleaning
    elif analysis_type == "data_processing":
        return "data_cleaner"  # Continue to cleaning then processing
    elif analysis_type == "tax_categorization":
        return "data_cleaner"  # Continue full workflow
    else:  # financial_analysis or strategic_planning
        return "data_cleaner"

# Add edges
graph_builder.add_edge(START, "data_fetcher")
graph_builder.add_conditional_edges(
    "data_fetcher",
    route_workflow,
    {
        "data_cleaner": "data_cleaner",
        END: END
    }
)

# Route from data_cleaner based on analysis_type
def route_from_cleaner(state: OverallState) -> str:
    """Route from data_cleaner based on analysis_type"""
    analysis_type = state.get("analysis_type", "financial_analysis")
    
    if analysis_type == "data_cleaning":
        return END  # Stop after cleaning
    else:
        return "data_processor"  # Continue to processing

graph_builder.add_conditional_edges(
    "data_cleaner",
    route_from_cleaner,
    {
        "data_processor": "data_processor",
        END: END
    }
)
graph_builder.add_edge("data_processor", "tax_categorizer")
graph_builder.add_edge("tax_categorizer", "report_generator")
graph_builder.add_edge("report_generator", END)

# Compile with checkpointer and store
checkpointer = InMemorySaver()
store = InMemoryStore()

app = graph_builder.compile(
    checkpointer=checkpointer,
    store=store
)
```

### 2.2 Agent Implementation Stubs

**Create placeholder agents with proper structure:**

**`src/agents/data_cleaner.py`:**
```python
from typing import Dict, Any, List, Tuple, Optional
from src.workflows.state_schemas import OverallState, Transaction
from src.utils.supabase_client import supabase_client
from src.utils.logging import StructuredLogger
from decimal import Decimal, InvalidOperation
from datetime import datetime, timedelta
import re
import pandas as pd
import numpy as np

logger = StructuredLogger()

def data_cleaner_agent(state: OverallState) -> Dict[str, Any]:
    """Clean, standardize, and format extracted data for optimal LLM processing"""
    trace_id = state.get("trace_id", "unknown")
    
    try:
        logger.log_agent_activity(
            "data_cleaner", "start_cleaning", trace_id
        )
        
        raw_data = state.get("raw_extracted_data", [])
        if not raw_data:
            return {"error_messages": ["No raw data to clean"]}
        
        # Comprehensive data cleaning pipeline
        cleaned_transactions, quality_metrics = comprehensive_data_cleaning(
            raw_data, trace_id
        )
        
        # Store cleaning results in local Supabase
        try:
            supabase_client.table("data_cleaning_logs").insert({
                "trace_id": trace_id,
                "client_id": state.get("client_id"),
                "raw_records_count": len(raw_data),
                "cleaned_records_count": len(cleaned_transactions),
                "quality_score": quality_metrics["overall_score"],
                "cleaning_summary": quality_metrics["summary"],
                "cleaning_timestamp": datetime.now().isoformat()
            }).execute()
        except Exception as db_error:
            logger.log_agent_activity(
                "data_cleaner", "storage_failed", trace_id,
                error=str(db_error)
            )
        
        logger.log_agent_activity(
            "data_cleaner", "cleaning_complete", trace_id,
            cleaned_count=len(cleaned_transactions),
            quality_score=quality_metrics["overall_score"]
        )
        
        return {
            "transactions": cleaned_transactions,
            "data_quality_score": quality_metrics["overall_score"],
            "cleaning_summary": quality_metrics["summary"],
            "workflow_phase": "data_cleaning_complete",
            "error_messages": []
        }
        
    except Exception as e:
        logger.log_agent_activity(
            "data_cleaner", "cleaning_failed", trace_id,
            error=str(e)
        )
        return {
            "transactions": [],
            "error_messages": [f"Data cleaner error: {str(e)}"],
            "workflow_phase": "data_cleaning_failed"
        }

def comprehensive_data_cleaning(raw_data: List[Dict], trace_id: str) -> Tuple[List[Transaction], Dict]:
    """Comprehensive data cleaning pipeline"""
    
    cleaning_stats = {
        "original_count": len(raw_data),
        "date_corrections": 0,
        "amount_corrections": 0,
        "description_cleanings": 0,
        "account_standardizations": 0,
        "duplicates_removed": 0,
        "invalid_records_dropped": 0
    }
    
    # Step 1: Convert to DataFrame for easier manipulation
    df = pd.DataFrame(raw_data)
    
    # Step 2: Standardize column names
    df = standardize_column_names(df)
    
    # Step 3: Clean and standardize dates
    df, date_corrections = clean_dates(df)
    cleaning_stats["date_corrections"] = date_corrections
    
    # Step 4: Clean and standardize amounts
    df, amount_corrections = clean_amounts(df)
    cleaning_stats["amount_corrections"] = amount_corrections
    
    # Step 5: Clean and enhance descriptions
    df, desc_cleanings = clean_descriptions(df)
    cleaning_stats["description_cleanings"] = desc_cleanings
    
    # Step 6: Standardize account information
    df, account_standardizations = standardize_accounts(df)
    cleaning_stats["account_standardizations"] = account_standardizations
    
    # Step 7: Remove duplicates
    df, duplicates_removed = remove_duplicates(df)
    cleaning_stats["duplicates_removed"] = duplicates_removed
    
    # Step 8: Validate and filter invalid records
    df, invalid_dropped = filter_invalid_records(df)
    cleaning_stats["invalid_records_dropped"] = invalid_dropped
    
    # Step 9: Extract additional metadata
    df = extract_metadata(df)
    
    # Step 10: Convert to Transaction objects
    transactions = convert_to_transactions(df)
    
    # Calculate quality metrics
    quality_metrics = calculate_quality_metrics(cleaning_stats, transactions)
    
    return transactions, quality_metrics

def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names across different file formats"""
    
    # Common column mapping patterns
    column_mappings = {
        # Date variations
        'transaction_date': 'date',
        'posted_date': 'date',
        'trans_date': 'date',
        'posting_date': 'date',
        
        # Description variations
        'transaction_description': 'description',
        'desc': 'description',
        'merchant': 'description',
        'memo': 'description',
        'details': 'description',
        
        # Amount variations
        'transaction_amount': 'amount',
        'debit': 'amount',
        'credit': 'amount',
        'withdrawal': 'amount',
        'deposit': 'amount',
        
        # Account variations
        'account_number': 'account',
        'account_id': 'account',
        'acct_num': 'account',
        
        # Category variations
        'transaction_category': 'category',
        'cat': 'category',
        'type': 'category'
    }
    
    # Apply mappings (case-insensitive)
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    df = df.rename(columns=column_mappings)
    
    return df

def clean_dates(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """Clean and standardize date formats"""
    corrections = 0
    
    if 'date' not in df.columns:
        return df, corrections
    
    # Try multiple date parsing strategies
    date_formats = [
        '%Y-%m-%d',
        '%m/%d/%Y',
        '%d/%m/%Y',
        '%m-%d-%Y',
        '%d-%m-%Y',
        '%Y/%m/%d',
        '%m/%d/%y',
        '%d/%m/%y'
    ]
    
    def parse_date(date_str):
        if pd.isna(date_str):
            return None
            
        date_str = str(date_str).strip()
        
        # Try each format
        for fmt in date_formats:
            try:
                return datetime.strptime(date_str, fmt).strftime('%Y-%m-%d')
            except ValueError:
                continue
        
        # Try pandas auto-parsing as last resort
        try:
            return pd.to_datetime(date_str).strftime('%Y-%m-%d')
        except:
            return None
    
    original_dates = df['date'].copy()
    df['date'] = df['date'].apply(parse_date)
    
    # Count corrections
    corrections = sum(1 for orig, new in zip(original_dates, df['date']) 
                     if orig != new and new is not None)
    
    return df, corrections

def clean_amounts(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """Clean and standardize amount formats"""
    corrections = 0
    
    if 'amount' not in df.columns:
        return df, corrections
    
    def parse_amount(amount_str):
        if pd.isna(amount_str):
            return None
            
        amount_str = str(amount_str).strip()
        
        # Remove currency symbols and formatting
        amount_str = re.sub(r'[^\d.-]', '', amount_str)
        
        # Handle parentheses (negative amounts)
        if '(' in str(amount_str) and ')' in str(amount_str):
            amount_str = '-' + re.sub(r'[()]', '', amount_str)
        
        try:
            return float(amount_str)
        except (ValueError, TypeError):
            return None
    
    original_amounts = df['amount'].copy()
    df['amount'] = df['amount'].apply(parse_amount)
    
    # Count corrections
    corrections = sum(1 for orig, new in zip(original_amounts, df['amount']) 
                     if orig != new and new is not None)
    
    return df, corrections

def clean_descriptions(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """Clean and enhance transaction descriptions"""
    cleanings = 0
    
    if 'description' not in df.columns:
        return df, cleanings
    
    def clean_description(desc):
        if pd.isna(desc):
            return "Unknown Transaction"
            
        desc = str(desc).strip()
        
        # Remove excessive whitespace
        desc = re.sub(r'\s+', ' ', desc)
        
        # Remove common bank codes and formatting
        desc = re.sub(r'\b\d{10,}\b', '', desc)  # Remove long numbers
        desc = re.sub(r'\*{3,}', '', desc)      # Remove asterisks
        desc = re.sub(r'#{3,}', '', desc)       # Remove hashes
        
        # Standardize common merchant patterns
        desc = re.sub(r'\bPOS\s+', '', desc, flags=re.IGNORECASE)
        desc = re.sub(r'\bATM\s+', 'ATM ', desc, flags=re.IGNORECASE)
        
        return desc.strip() or "Unknown Transaction"
    
    original_descriptions = df['description'].copy()
    df['description'] = df['description'].apply(clean_description)
    
    # Count cleanings
    cleanings = sum(1 for orig, new in zip(original_descriptions, df['description']) 
                   if orig != new)
    
    return df, cleanings

def standardize_accounts(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """Standardize account information"""
    standardizations = 0
    
    if 'account' not in df.columns:
        df['account'] = 'Unknown Account'
        return df, 0
    
    def standardize_account(account):
        if pd.isna(account):
            return "Unknown Account"
            
        account = str(account).strip()
        
        # Mask account numbers for privacy (show last 4 digits)
        if re.match(r'^\d{8,}$', account):
            return f"****{account[-4:]}"
        
        return account or "Unknown Account"
    
    original_accounts = df['account'].copy()
    df['account'] = df['account'].apply(standardize_account)
    
    # Count standardizations
    standardizations = sum(1 for orig, new in zip(original_accounts, df['account']) 
                          if orig != new)
    
    return df, standardizations

def remove_duplicates(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """Remove duplicate transactions"""
    original_count = len(df)
    
    # Define duplicate criteria (same date, amount, and similar description)
    df_deduped = df.drop_duplicates(
        subset=['date', 'amount', 'description'], 
        keep='first'
    )
    
    duplicates_removed = original_count - len(df_deduped)
    
    return df_deduped, duplicates_removed

def filter_invalid_records(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """Filter out invalid records"""
    original_count = len(df)
    
    # Remove records with missing critical data
    valid_df = df.dropna(subset=['date', 'amount', 'description'])
    
    # Remove records with zero amounts (configurable)
    # valid_df = valid_df[valid_df['amount'] != 0]
    
    invalid_dropped = original_count - len(valid_df)
    
    return valid_df, invalid_dropped

def extract_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Extract additional metadata for LLM processing"""
    
    # Add metadata columns that help LLM understanding
    df['transaction_type'] = df['amount'].apply(
        lambda x: 'income' if x > 0 else 'expense' if x < 0 else 'zero'
    )
    
    df['amount_magnitude'] = df['amount'].abs()
    
    # Extract potential vendor/merchant info
    df['potential_vendor'] = df['description'].apply(extract_vendor_name)
    
    # Add date components for temporal analysis
    df['month'] = pd.to_datetime(df['date']).dt.month
    df['day_of_week'] = pd.to_datetime(df['date']).dt.day_name()
    
    return df

def extract_vendor_name(description: str) -> str:
    """Extract likely vendor name from description"""
    if not description:
        return "Unknown"
    
    # Simple vendor extraction (can be enhanced)
    desc = description.upper()
    
    # Common patterns
    if 'AMAZON' in desc:
        return 'Amazon'
    elif 'WALMART' in desc:
        return 'Walmart'
    elif 'TARGET' in desc:
        return 'Target'
    elif 'STARBUCKS' in desc:
        return 'Starbucks'
    elif 'UBER' in desc:
        return 'Uber'
    elif 'LYFT' in desc:
        return 'Lyft'
    else:
        # Extract first meaningful word
        words = description.split()
        if words:
            return words[0][:20]  # Limit length
        return "Unknown"

def convert_to_transactions(df: pd.DataFrame) -> List[Transaction]:
    """Convert cleaned DataFrame to Transaction objects"""
    transactions = []
    
    for _, row in df.iterrows():
        try:
            transaction = Transaction(
                date=row.get('date'),
                description=row.get('description', 'Unknown'),
                amount=Decimal(str(row.get('amount', 0))),
                account=row.get('account', 'Unknown Account'),
                currency='USD',
                category=row.get('category'),
                # Add metadata for LLM context
                metadata={
                    'transaction_type': row.get('transaction_type'),
                    'potential_vendor': row.get('potential_vendor'),
                    'month': row.get('month'),
                    'day_of_week': row.get('day_of_week')
                }
            )
            transactions.append(transaction)
        except Exception as e:
            logger.log_agent_activity(
                "data_cleaner", "transaction_conversion_failed", "unknown",
                error=str(e), row_data=dict(row)
            )
            continue
    
    return transactions

def calculate_quality_metrics(stats: Dict, transactions: List[Transaction]) -> Dict:
    """Calculate data quality metrics"""
    
    total_corrections = (
        stats["date_corrections"] + 
        stats["amount_corrections"] + 
        stats["description_cleanings"] + 
        stats["account_standardizations"]
    )
    
    # Calculate overall quality score (0-100)
    if stats["original_count"] == 0:
        quality_score = 0
    else:
        retention_rate = len(transactions) / stats["original_count"]
        correction_rate = 1 - (total_corrections / stats["original_count"])
        quality_score = (retention_rate * 0.6 + correction_rate * 0.4) * 100
    
    return {
        "overall_score": round(quality_score, 2),
        "summary": {
            "records_processed": stats["original_count"],
            "records_retained": len(transactions),
            "retention_rate": round(len(transactions) / stats["original_count"] * 100, 2) if stats["original_count"] > 0 else 0,
            "corrections_made": total_corrections,
            "duplicates_removed": stats["duplicates_removed"],
            "invalid_dropped": stats["invalid_records_dropped"]
        }
    }
```
```python
from typing import Dict, Any, List
from src.workflows.state_schemas import OverallState, Transaction
from src.utils.file_processing import detect_file_type, parse_file
from src.utils.supabase_client import supabase_client
from src.utils.logging import StructuredLogger
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
                supabase_client.table("reports").insert({
                    "id": trace_id,
                    "client_id": client_id,
                    "analysis_type": state.get("analysis_type"),
                    "file_name": state.get("file_name", "uploaded_file"),
                    "processing_start_time": state.get("processing_start_time"),
                    "status": "processing",
                    "metadata": {
                        "file_type": file_type,
                        "transaction_count": len(transactions)
                    }
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
            "transactions": transactions,
            "workflow_phase": "data_extraction_complete",
            "error_messages": []
        }
        
    except Exception as e:
        logger.log_agent_activity(
            "data_fetcher", "extraction_failed", trace_id,
            error=str(e)
        )
        return {
            "transactions": [],
            "error_messages": [f"Data fetcher error: {str(e)}"],
            "workflow_phase": "data_extraction_failed"
        }
```

**`src/agents/data_processor.py`:**
```python
from typing import Dict, Any
from src.workflows.state_schemas import OverallState, FinancialMetrics
from src.utils.financial_calculations import calculate_metrics
from src.utils.supabase_client import supabase_client
from src.utils.logging import StructuredLogger

logger = StructuredLogger()

def data_processor_agent(state: OverallState) -> Dict[str, Any]:
    """Process and analyze financial transactions"""
    trace_id = state.get("trace_id", "unknown")
    
    try:
        logger.log_agent_activity(
            "data_processor", "start_processing", trace_id
        )
        
        transactions = state.get("transactions", [])
        if not transactions:
            return {"error_messages": ["No transactions to process"]}
        
        # Calculate financial metrics (placeholder)
        metrics = calculate_metrics(transactions)
        
        # Update report status in Supabase
        try:
            supabase_client.table("reports").update({
                "status": "data_processing_complete",
                "financial_metrics": metrics.dict() if hasattr(metrics, 'dict') else str(metrics)
            }).eq("id", trace_id).execute()
        except Exception as db_error:
            logger.log_agent_activity(
                "data_processor", "status_update_failed", trace_id,
                error=str(db_error)
            )
        
        logger.log_agent_activity(
            "data_processor", "processing_complete", trace_id,
            metrics_calculated=True
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
```

**`src/agents/report_generator.py`:**
```python
from typing import Dict, Any
from src.workflows.state_schemas import OverallState
from src.utils.supabase_client import supabase_client
from src.utils.logging import StructuredLogger
from src.config.settings import settings
import tempfile
import os

logger = StructuredLogger()

def report_generator_agent(state: OverallState) -> Dict[str, Any]:
    """Generate and store consulting-grade narrative report"""
    trace_id = state.get("trace_id", "unknown")
    
    try:
        logger.log_agent_activity(
            "report_generator", "start_generation", trace_id
        )
        
        # Generate placeholder report
        report_md = generate_placeholder_report(state)
        
        # Store report in Supabase Storage
        report_path = None
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as tmp_file:
                tmp_file.write(report_md)
                tmp_file_path = tmp_file.name
            
            # Upload to Supabase storage
            with open(tmp_file_path, 'rb') as file:
                client_id = state.get("client_id", "unknown")
                storage_path = f"{client_id}/{trace_id}/report.md"
                
                result = supabase_client.storage.from_(settings.storage_bucket).upload(
                    storage_path, file
                )
                
                if result.get('error'):
                    raise Exception(f"Storage upload failed: {result['error']}")
                
                report_path = storage_path
            
            # Clean up temp file
            os.unlink(tmp_file_path)
            
        except Exception as storage_error:
            logger.log_agent_activity(
                "report_generator", "storage_failed", trace_id,
                error=str(storage_error)
            )
        
        # Update final status in Supabase
        try:
            supabase_client.table("reports").update({
                "status": "completed",
                "report_path": report_path,
                "processing_end_time": "now()"
            }).eq("id", trace_id).execute()
        except Exception as db_error:
            logger.log_agent_activity(
                "report_generator", "final_status_update_failed", trace_id,
                error=str(db_error)
            )
        
        logger.log_agent_activity(
            "report_generator", "generation_complete", trace_id,
            report_stored=report_path is not None
        )
        
        return {
            "final_report_md": report_md,
            "final_report_pdf_path": report_path,
            "charts": [],
            "workflow_phase": "report_generation_complete",
            "error_messages": []
        }
        
    except Exception as e:
        logger.log_agent_activity(
            "report_generator", "generation_failed", trace_id,
            error=str(e)
        )
        return {
            "error_messages": [f"Report generator error: {str(e)}"],
            "workflow_phase": "report_generation_failed"
        }

def generate_placeholder_report(state: OverallState) -> str:
    """Generate a basic placeholder report"""
    transactions = state.get("transactions", [])
    financial_metrics = state.get("financial_metrics")
    
    report = f"""# Financial Analysis Report

## Executive Summary

Analysis completed for {len(transactions)} transactions.

## Key Findings

- Transactions processed: {len(transactions)}
- Analysis type: {state.get('analysis_type', 'financial_analysis')}
- Processing phase: {state.get('workflow_phase', 'unknown')}

## Recommendations

1. Continue monitoring financial performance
2. Implement regular analysis cycles
3. Consider automation opportunities

*Report generated by Ashworth Engine v2*
"""
    
    return report
```

### 2.3 Utility Functions Stubs

**Create placeholder utilities with proper interfaces:**

**`src/utils/file_processing.py`:**
```python
from typing import List, Any
import io
import pandas as pd
from src.workflows.state_schemas import Transaction

def detect_file_type(file_content: bytes) -> str:
    """Detect file type from content"""
    # Simple detection based on content
    if file_content.startswith(b'PK'):  # Excel files
        return "excel"
    elif b',' in file_content[:100]:  # CSV detection
        return "csv"
    elif b'%PDF' in file_content[:10]:
        return "pdf"
    else:
        return "unknown"

def parse_file(file_content: bytes, file_type: str) -> List[Transaction]:
    """Parse file content into transactions"""
    # Placeholder implementation
    if file_type == "excel":
        return parse_excel(file_content)
    elif file_type == "csv":
        return parse_csv(file_content)
    elif file_type == "pdf":
        return parse_pdf(file_content)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")

def parse_excel(file_content: bytes) -> List[Transaction]:
    """Parse Excel file content"""
    # Placeholder - returns sample transaction
    return [
        Transaction(
            date="2024-01-01",
            description="Sample transaction",
            amount=100.00,
            currency="USD"
        )
    ]

def parse_csv(file_content: bytes) -> List[Transaction]:
    """Parse CSV file content"""
    # Placeholder implementation
    return []

def parse_pdf(file_content: bytes) -> List[Transaction]:
    """Parse PDF file content with OCR"""
    # Placeholder implementation
    return []

def find_columns(df: pd.DataFrame) -> Dict[str, str]:
    """Detect which columns contain date, amount, description"""
    # Placeholder column detection logic
    return {
        "date": "Date",
        "amount": "Amount", 
        "description": "Description"
    }
```

**`src/utils/financial_calculations.py`:**
```python
from typing import List
from decimal import Decimal
from src.workflows.state_schemas import Transaction, FinancialMetrics

def calculate_metrics(transactions: List[Transaction]) -> FinancialMetrics:
    """Calculate comprehensive financial metrics"""
    # Placeholder implementation
    total_revenue = Decimal('1000.00')
    total_expenses = Decimal('500.00')
    
    return FinancialMetrics(
        total_revenue=total_revenue,
        total_expenses=total_expenses,
        gross_profit=total_revenue - total_expenses,
        gross_margin_pct=50.0,
        expense_by_category={"office": Decimal('300.00')},
        anomalies=[],
        pattern_matches={"vendor_count": 5},
        detected_business_types=["consulting"]
    )
```

### 2.4 FastAPI Integration

**Create API routes in `src/api/routes.py`:**
```python
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
            
            # Reset file pointer and upload
            file.file.seek(0)
            result = supabase_client.storage.from_("reports").upload(
                file_path, file.file
            )
            
            if not result.get('error'):
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
        
        # Execute workflow
        result = await workflow_app.ainvoke(initial_state)
        
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
                "charts_generated": len(result.get("charts", []))
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
        result = supabase_client.table("reports").select("*").eq("id", report_id).execute()
        
        if not result.data:
            raise HTTPException(status_code=404, detail="Report not found")
        
        report_data = result.data[0]
        
        # Calculate progress based on status
        status_progress = {
            "initialized": 10,
            "data_extraction_complete": 30,
            "data_processing_complete": 60,
            "tax_categorization_complete": 80,
            "completed": 100,
            "error": 0
        }
        
        progress = status_progress.get(report_data.get("status", "unknown"), 0)
        
        return {
            "report_id": report_id,
            "status": report_data.get("status"),
            "progress": progress,
            "created_at": report_data.get("created_at"),
            "processing_start_time": report_data.get("processing_start_time"),
            "processing_end_time": report_data.get("processing_end_time"),
            "metadata": report_data.get("metadata", {})
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
        "timestamp": datetime.utcnow().isoformat()
    }
    
    # Test Supabase connectivity
    try:
        supabase_client.table("reports").select("id").limit(1).execute()
        health_status["supabase"] = "connected"
    except Exception as e:
        health_status["supabase"] = f"error: {str(e)}"
        health_status["status"] = "degraded"
    
    return health_status

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### 2.5 Logging System

**Create structured logging in `src/utils/logging.py`:**
```python
import json
import logging
from datetime import datetime
from typing import Any, Dict

class StructuredLogger:
    """Structured JSON logger for agent activities"""
    
    def __init__(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def log_agent_activity(self, agent_name: str, activity: str, 
                          trace_id: str, **kwargs):
        """Log agent activity with structured format"""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "trace_id": trace_id,
            "agent": agent_name,
            "activity": activity,
            **kwargs
        }
        self.logger.info(json.dumps(log_entry))
    
    def log_error(self, agent_name: str, error: str, trace_id: str, **kwargs):
        """Log error with context"""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "trace_id": trace_id,
            "agent": agent_name,
            "level": "ERROR",
            "error": error,
            **kwargs
        }
        self.logger.error(json.dumps(log_entry))
```

### 2.6 Configuration Management

**Update `src/config/settings.py` with Supabase configuration:**
```python
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # LLM Configuration
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    llm_provider: str = "openai"
    embedding_model: str = "text-embedding-ada-002"
    
    # Supabase Configuration
    supabase_url: str = "http://127.0.0.1:54321"
    supabase_anon_key: Optional[str] = None
    supabase_service_key: Optional[str] = None
    supabase_jwt_secret: Optional[str] = None
    
    # Database Configuration
    database_url: str = "postgresql://postgres:postgres@127.0.0.1:54322/postgres"
    vecs_connection_string: str = "postgresql://postgres:postgres@127.0.0.1:54322/postgres"
    
    # Storage Configuration
    storage_provider: str = "supabase"
    storage_bucket: str = "reports"
    charts_bucket: str = "charts"
    supabase_storage_url: str = "http://127.0.0.1:54321/storage/v1"
    
    # Vector Database Configuration
    vector_collection_name: str = "documents"
    vector_dimension: int = 1536
    vector_similarity_threshold: float = 0.8
    
    # Processing Configuration
    max_upload_size: int = 52428800  # 50MB
    report_retention_days: int = 90
    ocr_language: str = "eng"
    
    # Performance Configuration
    max_concurrent_requests: int = 5
    llm_timeout_seconds: int = 300
    
    # API Configuration
    api_auth_key: Optional[str] = None
    ae_env: str = "development"
    
    # Logging Configuration
    log_level: str = "INFO"
    log_format: str = "json"
    
    # Validation
    def __post_init__(self):
        if self.llm_provider == "openai" and not self.openai_api_key:
            raise ValueError("OpenAI API key required when using OpenAI provider")
        
        if self.storage_provider == "supabase" and not self.supabase_service_key:
            raise ValueError("Supabase service key required when using Supabase storage")
    
    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()
```

### 2.7 Basic Error Handling

**Implement error handling patterns across all agents:**
- Try/catch blocks in all agent functions
- Structured error logging with context
- Graceful degradation (continue workflow when possible)
- Clear error messages in API responses

### 2.8 Supabase Integration Testing

**Create Supabase connectivity tests:**
```python
# tests/test_supabase_integration.py
import pytest
from src.utils.supabase_client import supabase_client, vecs_client
from src.config.settings import settings

def test_supabase_connection():
    """Test basic Supabase connectivity"""
    try:
        # Test basic query
        result = supabase_client.table("reports").select("id").limit(1).execute()
        assert result is not None
    except Exception as e:
        pytest.skip(f"Supabase not available: {e}")

def test_vector_client_connection():
    """Test vecs client connectivity"""
    try:
        # Test vector client
        vx = vecs_client
        # Try to get or create a test collection
        collection = vx.get_or_create_collection(
            name="test_collection", 
            dimension=384
        )
        assert collection is not None
    except Exception as e:
        pytest.skip(f"Vector database not available: {e}")

def test_storage_bucket_access():
    """Test Supabase storage bucket access"""
    try:
        # Test bucket listing
        result = supabase_client.storage.list_buckets()
        bucket_names = [bucket['name'] for bucket in result]
        assert settings.storage_bucket in bucket_names
    except Exception as e:
        pytest.skip(f"Storage not available: {e}")
```

### 2.9 Initial Testing

**Create basic tests in `tests/` directory:**
- Test workflow execution with sample data
- Test API endpoints with mock files
- Test error handling scenarios
- Verify logging output format

## Phase 2 Acceptance Criteria

- [ ] End-to-end API call processes simple test file (CSV)
- [ ] **All six agents implemented as `@task` functions** (data_fetcher, data_cleaner, data_processor, tax_categorizer, report_generator, orchestrator)
- [ ] **Data cleaner agent successfully standardizes and formats extracted data**
- [ ] **Data quality metrics calculated and stored in local Supabase**
- [ ] LangGraph workflow executes agents in correct sequence
- [ ] System returns structured result with status 200
- [ ] Logging captures each agent's invocation with trace_id
- [ ] Code passes linting with <200 LOC per module
- [ ] Clear separation between API and workflow layers
- [ ] Basic error handling prevents system crashes
- [ ] Configuration management working via environment variables
- [ ] Placeholder functionality working for all agents
- [ ] **Local Supabase connectivity established and tested**
- [ ] Report metadata stored in Supabase database
- [ ] File uploads stored in Supabase Storage
- [ ] **Data cleaning logs stored in local Supabase**
- [ ] Health check endpoint validates Supabase connection
- [ ] Vector database client connected and operational
- [ ] **Transaction metadata includes LLM-optimized context fields**

## Next Steps

After Phase 2 completion, proceed to Phase 3: Advanced Analytics & Report Generation.

## RACI Matrix

**Responsible:** Solo Developer
**Accountable:** Solo Developer  
**Consulted:** AI assistant for coding help, stakeholders for workflow validation
**Informed:** Stakeholders get demo of working pipeline