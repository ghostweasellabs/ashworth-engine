"""Base data models for the Ashworth Engine."""

from datetime import datetime
from decimal import Decimal
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator
import re


class Transaction(BaseModel):
    """Financial transaction model with robust data handling."""
    
    id: str
    date: datetime
    amount: Decimal
    description: str
    account_id: Optional[str] = None  # May be missing in messy data
    counterparty: Optional[str] = None
    category: Optional[str] = None
    tax_category: Optional[str] = None
    source_file: str
    data_quality_score: float = Field(ge=0.0, le=1.0)  # 0.0-1.0 indicating data completeness/accuracy
    data_issues: List[str] = Field(default_factory=list)  # List of data quality issues found
    
    @validator('amount', pre=True)
    def parse_amount(cls, v):
        """Handle various amount formats from messy Excel data."""
        if isinstance(v, str):
            # Remove currency symbols, commas, parentheses for negatives
            cleaned = re.sub(r'[^\d.-]', '', v.replace('(', '-').replace(')', ''))
            return Decimal(cleaned) if cleaned else Decimal('0')
        return Decimal(str(v))
    
    @validator('date', pre=True)
    def parse_date(cls, v):
        """Handle various date formats from messy Excel data."""
        if isinstance(v, str):
            # Try multiple date formats
            for fmt in ['%m/%d/%Y', '%Y-%m-%d', '%d/%m/%Y', '%m-%d-%Y']:
                try:
                    return datetime.strptime(v, fmt)
                except ValueError:
                    continue
            raise ValueError(f"Unable to parse date: {v}")
        return v


class FinancialMetrics(BaseModel):
    """Aggregated financial metrics."""
    
    total_revenue: Decimal
    total_expenses: Decimal
    net_income: Decimal
    expense_categories: Dict[str, Decimal]
    tax_deductible_amount: Decimal


class WorkflowResult(BaseModel):
    """Complete workflow result."""
    
    workflow_id: str
    client_id: str
    transactions: List[Transaction]
    metrics: FinancialMetrics
    report: str
    compliance_notes: List[str]
    processing_time: float