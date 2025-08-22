from typing import Annotated, List, Dict, Any, Optional
from typing_extensions import TypedDict
from decimal import Decimal
from datetime import datetime
from pydantic import BaseModel
import operator

class Transaction(BaseModel):
    date: str  # ISO format
    description: str
    amount: Decimal
    category: Optional[str] = None
    tax_category: Optional[str] = None
    is_deductible: Optional[bool] = None
    account: Optional[str] = None
    currency: str = "USD"
    metadata: Optional[Dict[str, Any]] = None  # Additional context for LLM processing
    
class FinancialMetrics(BaseModel):
    total_revenue: Decimal
    total_expenses: Decimal
    gross_profit: Decimal
    gross_margin_pct: float
    cash_balance: Optional[Decimal] = None
    expense_by_category: Dict[str, Decimal]
    anomalies: List[Transaction]
    pattern_matches: Dict[str, int]
    detected_business_types: List[str]
    
class TaxSummary(BaseModel):
    total_deductible_expenses: Decimal
    total_taxable_income: Decimal
    business_expense_categories: Dict[str, Decimal]
    tax_optimization_suggestions: List[str]
    compliance_warnings: List[str]
    categorization_accuracy: float = 100.0
    
class OverallState(TypedDict):
    # Input data
    client_id: str
    analysis_type: str
    file_content: bytes
    
    # Processing state
    raw_extracted_data: Annotated[List[Dict[str, Any]], operator.add]  # Raw OCR/extraction output
    transactions: Annotated[List[Transaction], operator.add]  # Cleaned and structured data
    financial_metrics: Optional[FinancialMetrics]
    tax_summary: Optional[TaxSummary]
    
    # Data quality tracking
    data_quality_score: Optional[float]
    cleaning_summary: Optional[Dict[str, Any]]
    
    # Output state
    final_report_md: Optional[str]
    final_report_pdf_path: Optional[str]
    charts: List[str]
    
    # Error handling
    error_messages: Annotated[List[str], operator.add]
    warnings: Annotated[List[str], operator.add]
    
    # Metadata
    workflow_phase: str
    processing_start_time: datetime
    processing_end_time: Optional[datetime]
    trace_id: str

class InputState(TypedDict):
    client_id: str
    analysis_type: str
    file_content: bytes

class OutputState(TypedDict):
    final_report_md: str
    final_report_pdf_path: str
    charts: List[str]
    error_messages: List[str]from typing import Annotated, List, Dict, Any, Optional
from typing_extensions import TypedDict
from decimal import Decimal
from datetime import datetime
from pydantic import BaseModel
import operator

class Transaction(BaseModel):
    date: str  # ISO format
    description: str
    amount: Decimal
    category: Optional[str] = None
    tax_category: Optional[str] = None
    is_deductible: Optional[bool] = None
    account: Optional[str] = None
    currency: str = "USD"
    metadata: Optional[Dict[str, Any]] = None  # Additional context for LLM processing
    
class FinancialMetrics(BaseModel):
    total_revenue: Decimal
    total_expenses: Decimal
    gross_profit: Decimal
    gross_margin_pct: float
    cash_balance: Optional[Decimal] = None
    expense_by_category: Dict[str, Decimal]
    anomalies: List[Transaction]
    pattern_matches: Dict[str, int]
    detected_business_types: List[str]
    
class TaxSummary(BaseModel):
    deductible_total: Decimal
    non_deductible_total: Decimal
    potential_savings: Decimal
    flags: List[str]
    categorization_accuracy: float = 100.0
    
class OverallState(TypedDict):
    # Input data
    client_id: str
    analysis_type: str
    file_content: bytes
    
    # Processing state
    raw_extracted_data: Annotated[List[Dict[str, Any]], operator.add]  # Raw OCR/extraction output
    transactions: Annotated[List[Transaction], operator.add]  # Cleaned and structured data
    financial_metrics: Optional[FinancialMetrics]
    tax_summary: Optional[TaxSummary]
    
    # Data quality tracking
    data_quality_score: Optional[float]
    cleaning_summary: Optional[Dict[str, Any]]
    
    # Output state
    final_report_md: Optional[str]
    final_report_pdf_path: Optional[str]
    charts: List[str]
    
    # Error handling
    error_messages: Annotated[List[str], operator.add]
    warnings: Annotated[List[str], operator.add]
    
    # Metadata
    workflow_phase: str
    processing_start_time: datetime
    processing_end_time: Optional[datetime]
    trace_id: str

class InputState(TypedDict):
    client_id: str
    analysis_type: str
    file_content: bytes

class OutputState(TypedDict):
    final_report_md: str
    final_report_pdf_path: str
    charts: List[str]
    error_messages: List[str]