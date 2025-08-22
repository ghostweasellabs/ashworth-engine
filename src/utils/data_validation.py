"""Data validation utilities for financial data processing"""

from typing import Dict, List, Any, Optional
from decimal import Decimal, InvalidOperation
from datetime import datetime
import re

def validate_transaction_data(data: Dict[str, Any]) -> Dict[str, List[str]]:
    """Validate transaction data and return validation errors"""
    errors = {}
    
    # Required fields validation
    required_fields = ["date", "description", "amount"]
    for field in required_fields:
        if field not in data or data[field] is None:
            errors.setdefault("missing_fields", []).append(field)
    
    # Date validation
    if "date" in data and data["date"]:
        if not validate_date_format(data["date"]):
            errors.setdefault("date", []).append("Invalid date format. Expected ISO format (YYYY-MM-DD)")
    
    # Amount validation
    if "amount" in data and data["amount"] is not None:
        if not validate_amount(data["amount"]):
            errors.setdefault("amount", []).append("Invalid amount format")
    
    # Currency validation
    if "currency" in data and data["currency"]:
        if not validate_currency_code(data["currency"]):
            errors.setdefault("currency", []).append("Invalid currency code")
    
    return errors

def validate_date_format(date_str: str) -> bool:
    """Validate date string format (ISO format preferred)"""
    try:
        # Try ISO format first
        datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        return True
    except ValueError:
        # Try common formats
        formats = ["%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%Y-%m-%d %H:%M:%S"]
        for fmt in formats:
            try:
                datetime.strptime(date_str, fmt)
                return True
            except ValueError:
                continue
    return False

def validate_amount(amount: Any) -> bool:
    """Validate amount can be converted to Decimal"""
    try:
        if isinstance(amount, (int, float)):
            Decimal(str(amount))
            return True
        elif isinstance(amount, str):
            # Remove common formatting
            clean_amount = amount.replace(',', '').replace('$', '').replace('€', '').strip()
            Decimal(clean_amount)
            return True
        elif isinstance(amount, Decimal):
            return True
    except (InvalidOperation, ValueError):
        pass
    return False

# TODO: We're only going to use USD, no other currencies, remove and fix.
def validate_currency_code(currency: str) -> bool:
    """Validate currency code (basic validation)"""
    # Common currency codes
    common_currencies = {
        "USD", "EUR", "GBP", "JPY", "CAD", "AUD", "CHF", "CNY", "SEK", "NOK"
    }
    return currency.upper() in common_currencies or len(currency) == 3

def calculate_data_quality_score(data: List[Dict[str, Any]]) -> float:
    """Calculate overall data quality score (0-100)"""
    if not data:
        return 0.0
    
    total_score = 0.0
    
    for record in data:
        record_score = 0.0
        max_score = 0.0
        
        # Required fields completeness (40% weight)
        required_fields = ["date", "description", "amount"]
        for field in required_fields:
            max_score += 40 / len(required_fields)
            if field in record and record[field] is not None:
                if field == "date" and validate_date_format(str(record[field])):
                    record_score += 40 / len(required_fields)
                elif field == "amount" and validate_amount(record[field]):
                    record_score += 40 / len(required_fields)
                elif field == "description" and len(str(record[field]).strip()) > 0:
                    record_score += 40 / len(required_fields)
        
        # Optional fields completeness (30% weight)
        optional_fields = ["category", "account", "currency"]
        for field in optional_fields:
            max_score += 30 / len(optional_fields)
            if field in record and record[field] is not None and len(str(record[field]).strip()) > 0:
                record_score += 30 / len(optional_fields)
        
        # Data format consistency (30% weight)
        max_score += 30
        format_score = 0
        if "date" in record and validate_date_format(str(record["date"])):
            format_score += 10
        if "amount" in record and validate_amount(record["amount"]):
            format_score += 10
        if "currency" in record and validate_currency_code(str(record["currency"])):
            format_score += 10
        record_score += format_score
        
        total_score += (record_score / max_score) * 100 if max_score > 0 else 0
    
    return total_score / len(data) if data else 0.0