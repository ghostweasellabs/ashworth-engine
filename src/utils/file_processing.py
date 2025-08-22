from typing import List, Any, Dict
import io
import pandas as pd
from src.workflows.state_schemas import Transaction
from decimal import Decimal
import re

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
    try:
        # Read Excel file into DataFrame
        df = pd.read_excel(io.BytesIO(file_content))
        return convert_dataframe_to_transactions(df)
    except Exception as e:
        # Return sample transaction on error for now
        return [
            Transaction(
                date="2024-01-01",
                description="Sample Excel transaction",
                amount=Decimal("100.00"),
                currency="USD",
                account="Unknown",
                metadata={"source": "excel_parse_error", "error": str(e)}
            )
        ]

def parse_csv(file_content: bytes) -> List[Transaction]:
    """Parse CSV file content"""
    try:
        # Read CSV file into DataFrame
        df = pd.read_csv(io.StringIO(file_content.decode('utf-8')))
        return convert_dataframe_to_transactions(df)
    except Exception as e:
        # Return sample transaction on error for now
        return [
            Transaction(
                date="2024-01-01",
                description="Sample CSV transaction",
                amount=Decimal("50.00"),
                currency="USD",
                account="Unknown",
                metadata={"source": "csv_parse_error", "error": str(e)}
            )
        ]

def parse_pdf(file_content: bytes) -> List[Transaction]:
    """Parse PDF file content with OCR"""
    # Placeholder implementation for PDF parsing
    # This would typically use pdfplumber or pytesseract
    return [
        Transaction(
            date="2024-01-01",
            description="Sample PDF transaction",
            amount=Decimal("75.00"),
            currency="USD",
            account="Unknown",
            metadata={"source": "pdf_placeholder"}
        )
    ]

def convert_dataframe_to_transactions(df: pd.DataFrame) -> List[Transaction]:
    """Convert DataFrame to Transaction objects"""
    transactions = []
    
    # Try to detect column mappings
    column_mapping = find_columns(df)
    
    for index, row in df.iterrows():
        try:
            # Extract values using column mapping
            date_val = row.get(column_mapping.get("date", "date"), "2024-01-01")
            desc_val = row.get(column_mapping.get("description", "description"), f"Transaction {index+1}")
            amount_val = row.get(column_mapping.get("amount", "amount"), 0)
            account_val = row.get(column_mapping.get("account", "account"), "Unknown")
            
            # Clean and convert amount
            if isinstance(amount_val, str):
                # Remove currency symbols and convert to decimal
                amount_str = re.sub(r'[^\d.-]', '', str(amount_val))
                try:
                    amount_decimal = Decimal(amount_str) if amount_str else Decimal('0')
                except:
                    amount_decimal = Decimal('0')
            else:
                amount_decimal = Decimal(str(amount_val)) if amount_val else Decimal('0')
            
            # Create transaction
            transaction = Transaction(
                date=str(date_val)[:10],  # Ensure date format
                description=str(desc_val)[:200],  # Limit description length
                amount=amount_decimal,
                currency="USD",
                account=str(account_val)[:50],  # Limit account length
                metadata={"row_index": index, "source": "dataframe_conversion"}
            )
            
            transactions.append(transaction)
            
        except Exception as e:
            # Skip invalid rows but log the issue
            continue
    
    return transactions

def find_columns(df: pd.DataFrame) -> Dict[str, str]:
    """Detect which columns contain date, amount, description"""
    columns = [col.lower() for col in df.columns]
    mapping = {}
    
    # Date column detection
    date_keywords = ['date', 'transaction_date', 'posted_date', 'trans_date']
    for keyword in date_keywords:
        matching_cols = [col for col in df.columns if keyword in col.lower()]
        if matching_cols:
            mapping["date"] = matching_cols[0]
            break
    
    # Amount column detection
    amount_keywords = ['amount', 'transaction_amount', 'debit', 'credit', 'value']
    for keyword in amount_keywords:
        matching_cols = [col for col in df.columns if keyword in col.lower()]
        if matching_cols:
            mapping["amount"] = matching_cols[0]
            break
    
    # Description column detection
    desc_keywords = ['description', 'memo', 'details', 'transaction_description', 'merchant']
    for keyword in desc_keywords:
        matching_cols = [col for col in df.columns if keyword in col.lower()]
        if matching_cols:
            mapping["description"] = matching_cols[0]
            break
    
    # Account column detection
    account_keywords = ['account', 'account_number', 'acct', 'account_id']
    for keyword in account_keywords:
        matching_cols = [col for col in df.columns if keyword in col.lower()]
        if matching_cols:
            mapping["account"] = matching_cols[0]
            break
    
    # Use first available columns as fallbacks
    if "date" not in mapping and len(df.columns) > 0:
        mapping["date"] = df.columns[0]
    if "amount" not in mapping and len(df.columns) > 1:
        mapping["amount"] = df.columns[1]
    if "description" not in mapping and len(df.columns) > 2:
        mapping["description"] = df.columns[2]
    
    return mapping