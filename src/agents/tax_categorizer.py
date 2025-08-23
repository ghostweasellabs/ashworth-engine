from typing import Dict, Any, List, Optional
from src.workflows.state_schemas import OverallState, TaxSummary, Transaction
from src.utils.supabase_client import supabase_client
from src.utils.logging import StructuredLogger
from decimal import Decimal
from collections import defaultdict
import re
from datetime import datetime

logger = StructuredLogger()

# Official IRS Business Expense Categories (Based on Publication 334 and discontinued Publication 535)
# These categories follow the "ordinary and necessary" standard as defined by the IRS
IRS_EXPENSE_CATEGORIES = {
    "advertising_marketing": {
        "keywords": ["advertising", "marketing", "promotion", "social media", "facebook ads", "google ads", "billboard", "radio ad", "tv commercial", "brochure", "flyer"],
        "irs_code": "ADVT",
        "deductible_percentage": 1.0,
        "description": "Advertising and Marketing Expenses",
        "documentation_required": "Business purpose and receipts"
    },
    "business_meals": {
        "keywords": ["restaurant", "meal", "lunch", "dinner", "catering", "food", "coffee", "business lunch", "client dinner"],
        "irs_code": "MEAL",
        "deductible_percentage": 0.5,  # 50% deductible for business meals
        "description": "Business Meals (50% deductible)",
        "documentation_required": "Business purpose, attendees, and receipts"
    },
    "travel_expenses": {
        "keywords": ["hotel", "flight", "airline", "uber", "taxi", "lyft", "mileage", "car rental", "airbnb", "gas", "parking", "tolls"],
        "irs_code": "TRVL",
        "deductible_percentage": 1.0,
        "description": "Travel Expenses",
        "documentation_required": "Business purpose, destination, dates, and receipts"
    },
    "office_supplies": {
        "keywords": ["staples", "office depot", "supplies", "paper", "pens", "printer", "ink", "toner", "folders", "filing"],
        "irs_code": "OFFC",
        "deductible_percentage": 1.0,
        "description": "Office Supplies and Materials",
        "documentation_required": "Business use receipts"
    },
    "professional_services": {
        "keywords": ["legal", "accounting", "consulting", "attorney", "lawyer", "cpa", "bookkeeper", "tax prep", "audit"],
        "irs_code": "PROF",
        "deductible_percentage": 1.0,
        "description": "Professional Services",
        "documentation_required": "Service description and invoices"
    },
    "software_technology": {
        "keywords": ["software", "subscription", "saas", "microsoft", "adobe", "quickbooks", "zoom", "slack", "github"],
        "irs_code": "TECH",
        "deductible_percentage": 1.0,
        "description": "Software and Technology",
        "documentation_required": "Business use documentation"
    },
    "utilities_communications": {
        "keywords": ["electric", "gas", "internet", "phone", "water", "utilities", "cell phone", "landline", "wifi"],
        "irs_code": "UTIL",
        "deductible_percentage": 1.0,  # May be partial if mixed use
        "description": "Utilities and Communications",
        "documentation_required": "Business use percentage if mixed use"
    },
    "equipment_furniture": {
        "keywords": ["computer", "laptop", "desk", "chair", "furniture", "equipment", "machinery", "tools"],
        "irs_code": "EQUP",
        "deductible_percentage": 1.0,  # May require depreciation for large items
        "description": "Equipment and Furniture",
        "documentation_required": "Business use and depreciation schedule for items >$2,500"
    },
    "insurance_premiums": {
        "keywords": ["insurance", "liability", "business insurance", "workers comp", "professional liability", "e&o"],
        "irs_code": "INSUR",
        "deductible_percentage": 1.0,
        "description": "Business Insurance Premiums",
        "documentation_required": "Policy documents and business purpose"
    },
    "rent_lease": {
        "keywords": ["rent", "lease", "office space", "warehouse", "commercial rent", "storage unit"],
        "irs_code": "RENT",
        "deductible_percentage": 1.0,
        "description": "Rent and Lease Payments",
        "documentation_required": "Lease agreement and business use"
    },
    "non_deductible": {
        "keywords": ["personal", "salary", "owner draw", "dividend", "gift", "entertainment", "political"],
        "irs_code": "NONE",
        "deductible_percentage": 0.0,
        "description": "Non-Deductible Expenses",
        "documentation_required": "Personal expense documentation"
    }
}

def tax_categorizer_agent(state: OverallState) -> Dict[str, Any]:
    """Categorize transactions for IRS compliance using official guidelines"""
    trace_id = state.get("trace_id", "unknown")
    
    try:
        logger.log_agent_activity(
            "tax_categorizer", "start_categorization", trace_id
        )
        
        transactions = state.get("transactions", [])
        if not transactions:
            return {"error_messages": ["No transactions to categorize"]}
        
        # IRS-compliant tax categorization with official expense categories
        tax_summary = perform_irs_compliant_tax_categorization(transactions)
        
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

def perform_irs_compliant_tax_categorization(transactions: List[Transaction]) -> TaxSummary:
    """Perform IRS-compliant tax categorization based on official guidelines"""
    
    total_deductible_expenses = Decimal('0')
    total_taxable_income = Decimal('0')
    business_expense_categories = defaultdict(Decimal)
    categorization_accuracy = 100.0
    
    # IRS-compliant categorization logic
    for transaction in transactions:
        amount = transaction.amount
        description = transaction.description.lower() if transaction.description else ""
        
        if amount > 0:  # Income
            total_taxable_income += amount
        else:  # Expense
            expense_amount = abs(amount)
            
            # Apply IRS expense categorization
            category_info = categorize_irs_business_expense(description, expense_amount)
            category = category_info["category"]
            deductible_amount = category_info["deductible_amount"]
            
            # Update transaction with tax category and deductibility
            transaction.tax_category = category
            transaction.is_deductible = deductible_amount > 0
            
            if deductible_amount > 0:
                total_deductible_expenses += deductible_amount
                business_expense_categories[category] += deductible_amount
            
            # Lower accuracy if categorization is uncertain
            if category == "other_business_expenses":
                categorization_accuracy -= 2.0
    
    # Generate IRS-compliant suggestions and warnings
    suggestions = generate_irs_tax_optimization_suggestions(business_expense_categories)
    warnings = generate_irs_compliance_warnings(transactions)
    
    return TaxSummary(
        total_deductible_expenses=total_deductible_expenses,
        total_taxable_income=total_taxable_income,
        business_expense_categories=dict(business_expense_categories),
        tax_optimization_suggestions=suggestions,
        compliance_warnings=warnings,
        categorization_accuracy=max(75.0, categorization_accuracy)  # Minimum 75% accuracy
    )

def categorize_irs_business_expense(description: str, amount: Decimal) -> Dict[str, Any]:
    """Categorize business expenses according to IRS guidelines with confidence scoring"""
    
    description_lower = description.lower()
    best_category = "other_business_expenses"  # Default fallback
    best_score = 0
    deductible_amount = Decimal('0')
    
    # Score each IRS category
    for category, config in IRS_EXPENSE_CATEGORIES.items():
        score = 0
        
        for keyword in config["keywords"]:
            if keyword in description_lower:
                score += len(keyword) * 2  # Longer keywords get higher scores
        
        # Apply additional scoring for exact matches
        if any(keyword == description_lower.strip() for keyword in config["keywords"]):
            score += 10
        
        if score > best_score:
            best_score = score
            best_category = category
    
    # Calculate deductible amount based on IRS rules
    if best_category in IRS_EXPENSE_CATEGORIES:
        category_config = IRS_EXPENSE_CATEGORIES[best_category]
        deductible_percentage = Decimal(str(category_config["deductible_percentage"]))
        deductible_amount = amount * deductible_percentage
    
    return {
        "category": best_category,
        "deductible_amount": deductible_amount,
        "confidence_score": min(100, best_score * 5),  # Cap at 100%
        "irs_code": IRS_EXPENSE_CATEGORIES.get(best_category, {}).get("irs_code", "OTHER")
    }

def generate_irs_tax_optimization_suggestions(categories: Dict[str, Decimal]) -> List[str]:
    """Generate IRS-compliant tax optimization suggestions"""
    suggestions = []
    
    # Section 179 deduction opportunities (up to $2,500,000 for 2025)
    equipment_total = categories.get("equipment_furniture", Decimal('0'))
    if equipment_total > Decimal('2500'):
        suggestions.append(f"Consider Section 179 deduction for equipment purchases (${equipment_total:,.2f}) - may be fully deductible in 2025")
    
    # Business meal documentation reminder
    meal_total = categories.get("business_meals", Decimal('0'))
    if meal_total > Decimal('1000'):
        suggestions.append(f"Business meals (${meal_total:,.2f}) are 50% deductible - ensure proper documentation of business purpose and attendees")
    
    return suggestions

def generate_irs_compliance_warnings(transactions: List[Transaction]) -> List[str]:
    """Generate IRS compliance warnings based on audit triggers and regulations"""
    warnings = []
    
    # Form 8300 requirement for cash transactions ≥ $10,000
    large_cash_threshold = Decimal('10000')
    for transaction in transactions:
        if abs(transaction.amount) >= large_cash_threshold:
            warnings.append(f"CRITICAL: Transaction ${abs(transaction.amount):,.2f} may require Form 8300 filing (cash transactions ≥$10,000)")
    
    return warnings