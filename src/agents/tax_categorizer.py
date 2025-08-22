from typing import Dict, Any, List
from src.workflows.state_schemas import OverallState, TaxSummary, Transaction
from src.utils.supabase_client import supabase_client
from src.utils.logging import StructuredLogger
from decimal import Decimal
from collections import defaultdict

logger = StructuredLogger()

def tax_categorizer_agent(state: OverallState) -> Dict[str, Any]:
    """Categorize transactions for tax compliance (Phase 2 stub implementation)"""
    trace_id = state.get("trace_id", "unknown")
    
    try:
        logger.log_agent_activity(
            "tax_categorizer", "start_categorization", trace_id
        )
        
        transactions = state.get("transactions", [])
        if not transactions:
            return {"error_messages": ["No transactions to categorize"]}
        
        # Basic tax categorization (placeholder for Phase 3 enhancement)
        tax_summary = perform_basic_tax_categorization(transactions)
        
        # Update analysis status in Supabase
        try:
            client_id = state.get("client_id")
            if client_id:
                # Convert tax summary to dict for JSON storage
                tax_dict = {
                    "total_deductible_expenses": float(tax_summary.total_deductible_expenses),
                    "total_taxable_income": float(tax_summary.total_taxable_income),
                    "business_expense_categories": {k: float(v) for k, v in tax_summary.business_expense_categories.items()},
                    "tax_optimization_suggestions": tax_summary.tax_optimization_suggestions,
                    "compliance_warnings": tax_summary.compliance_warnings
                }
                
                supabase_client.table("analyses").update({
                    "status": "tax_categorization_complete",
                    "results": {
                        **state.get("results", {}),
                        "tax_summary": tax_dict
                    }
                }).eq("id", trace_id).execute()
        except Exception as db_error:
            logger.log_agent_activity(
                "tax_categorizer", "status_update_failed", trace_id,
                error=str(db_error)
            )
        
        logger.log_agent_activity(
            "tax_categorizer", "categorization_complete", trace_id,
            deductible_expenses=float(tax_summary.total_deductible_expenses),
            taxable_income=float(tax_summary.total_taxable_income)
        )
        
        return {
            "tax_summary": tax_summary,
            "workflow_phase": "tax_categorization_complete",
            "error_messages": []
        }
        
    except Exception as e:
        logger.log_agent_activity(
            "tax_categorizer", "categorization_failed", trace_id,
            error=str(e)
        )
        return {
            "error_messages": [f"Tax categorizer error: {str(e)}"],
            "workflow_phase": "tax_categorization_failed"
        }

def perform_basic_tax_categorization(transactions: List[Transaction]) -> TaxSummary:
    """Perform basic tax categorization (Phase 2 placeholder implementation)"""
    
    total_deductible_expenses = Decimal('0')
    total_taxable_income = Decimal('0')
    business_expense_categories = defaultdict(Decimal)
    
    # Basic categorization logic
    for transaction in transactions:
        amount = transaction.amount
        description = transaction.description.lower() if transaction.description else ""
        
        if amount > 0:  # Income
            total_taxable_income += amount
        else:  # Expense
            expense_amount = abs(amount)
            
            # Basic business expense detection
            if is_business_expense(description):
                total_deductible_expenses += expense_amount
                category = categorize_business_expense(description)
                business_expense_categories[category] += expense_amount
    
    # Generate basic suggestions and warnings
    suggestions = generate_tax_optimization_suggestions(business_expense_categories)
    warnings = generate_compliance_warnings(transactions)
    
    return TaxSummary(
        total_deductible_expenses=total_deductible_expenses,
        total_taxable_income=total_taxable_income,
        business_expense_categories=dict(business_expense_categories),
        tax_optimization_suggestions=suggestions,
        compliance_warnings=warnings
    )

def is_business_expense(description: str) -> bool:
    """Basic business expense detection"""
    business_keywords = [
        'office', 'supply', 'software', 'subscription', 'professional',
        'consulting', 'marketing', 'advertising', 'travel', 'meal',
        'hotel', 'conference', 'training', 'equipment', 'internet',
        'phone', 'utilities', 'rent', 'insurance', 'legal', 'accounting'
    ]
    
    return any(keyword in description for keyword in business_keywords)

def categorize_business_expense(description: str) -> str:
    """Categorize business expenses for tax purposes"""
    
    # Office supplies and equipment
    if any(keyword in description for keyword in ['office', 'supply', 'equipment', 'furniture']):
        return "office_equipment"
    
    # Professional services
    if any(keyword in description for keyword in ['professional', 'consulting', 'legal', 'accounting']):
        return "professional_services"
    
    # Marketing and advertising
    if any(keyword in description for keyword in ['marketing', 'advertising', 'promotion']):
        return "marketing_advertising"
    
    # Travel and meals
    if any(keyword in description for keyword in ['travel', 'hotel', 'meal', 'restaurant', 'uber', 'lyft']):
        return "travel_meals"
    
    # Software and subscriptions
    if any(keyword in description for keyword in ['software', 'subscription', 'saas']):
        return "software_subscriptions"
    
    # Utilities and communications
    if any(keyword in description for keyword in ['internet', 'phone', 'utilities']):
        return "utilities_communications"
    
    return "other_business_expenses"

def generate_tax_optimization_suggestions(categories: Dict[str, Decimal]) -> List[str]:
    """Generate basic tax optimization suggestions"""
    suggestions = []
    
    # Check for missing common deductions
    if "professional_services" not in categories or categories["professional_services"] < Decimal('1000'):
        suggestions.append("Consider tracking professional service expenses for tax deductions")
    
    if "software_subscriptions" in categories and categories["software_subscriptions"] > Decimal('500'):
        suggestions.append("Software subscriptions may be fully deductible as business expenses")
    
    if "travel_meals" in categories:
        suggestions.append("Business meals may be 50% deductible - ensure proper documentation")
    
    return suggestions

def generate_compliance_warnings(transactions: List[Transaction]) -> List[str]:
    """Generate basic compliance warnings"""
    warnings = []
    
    # Check for large cash transactions
    large_cash_threshold = Decimal('10000')
    for transaction in transactions:
        if abs(transaction.amount) > large_cash_threshold:
            warnings.append(f"Large transaction detected: ${abs(transaction.amount)} - ensure proper documentation")
    
    # Check for round number patterns (potential manual entries)
    round_numbers = [t for t in transactions if abs(t.amount) % 100 == 0 and abs(t.amount) >= 100]
    if len(round_numbers) > len(transactions) * 0.3:
        warnings.append("High percentage of round-number transactions - review for accuracy")
    
    return warnings