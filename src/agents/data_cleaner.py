from typing import Dict, Any, List, Tuple, Optional
from langchain_core.runnables import RunnableConfig
from src.workflows.state_schemas import OverallState, Transaction
from src.utils.supabase_client import supabase_client
from src.utils.logging import StructuredLogger
from decimal import Decimal, InvalidOperation
from datetime import datetime, timedelta
import re
import pandas as pd
import numpy as np

logger = StructuredLogger()

def data_cleaner_agent(state: OverallState, 
                      config: Optional[RunnableConfig] = None, 
                      *, 
                      store=None) -> Dict[str, Any]:
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
            # Note: This table needs to be created in migration
            # For now, we'll store in analyses table with cleaning info
            client_id = state.get("client_id")
            if client_id:
                supabase_client.table("analyses").update({
                    "status": "data_cleaning_complete",
                    "results": {
                        "raw_records_count": len(raw_data),
                        "cleaned_records_count": len(cleaned_transactions),
                        "quality_score": quality_metrics["overall_score"],
                        "cleaning_summary": quality_metrics["summary"]
                    }
                }).eq("id", trace_id).execute()
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
                date=row.get('date') or '2024-01-01',
                description=row.get('description') or 'Unknown',
                amount=Decimal(str(row.get('amount', 0))),
                account=row.get('account') or 'Unknown Account',
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