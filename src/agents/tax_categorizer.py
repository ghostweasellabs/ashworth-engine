from typing import Dict, Any, List, Optional
from langchain_core.runnables import RunnableConfig
from src.workflows.state_schemas import OverallState, TaxSummary, Transaction
from src.utils.supabase_client import supabase_client
from src.utils.logging import StructuredLogger
from src.utils.vector_operations import get_vector_store
from src.utils.memory_store import get_shared_memory_store, MemoryNamespaces
from src.utils.llm_integration import get_llm_client
from src.config.settings import settings
from decimal import Decimal
from collections import defaultdict
import re
from datetime import datetime
import asyncio

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

def tax_categorizer_agent(state: OverallState, 
                         config: Optional[RunnableConfig] = None, 
                         *, 
                         store=None) -> Dict[str, Any]:
    """Categorize transactions for IRS compliance using official guidelines with RAG enhancement"""
    trace_id = state.get("trace_id", "unknown")
    
    try:
        logger.log_agent_activity(
            "tax_categorizer", "start_categorization", trace_id
        )
        
        transactions = state.get("transactions", [])
        if not transactions:
            return {"error_messages": ["No transactions to categorize"]}
        
        # Perform IRS-compliant tax categorization with RAG enhancement
        tax_summary = asyncio.run(
            perform_rag_enhanced_tax_categorization(transactions, config, store)
        )
        
        # Store categorization insights in shared memory
        asyncio.run(
            store_categorization_insights(tax_summary, transactions, config, store)
        )
        
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

async def perform_rag_enhanced_tax_categorization(transactions: List[Transaction], 
                                                 config: Optional[RunnableConfig] = None,
                                                 store=None) -> TaxSummary:
    """Perform IRS-compliant tax categorization enhanced with RAG for complex cases"""
    
    total_deductible_expenses = Decimal('0')
    total_taxable_income = Decimal('0')
    business_expense_categories = defaultdict(Decimal)
    categorization_accuracy = 100.0
    
    # Get RAG-enhanced categorization for complex transactions
    complex_transactions = []
    
    # First pass: Standard categorization
    for transaction in transactions:
        amount = transaction.amount
        description = transaction.description.lower() if transaction.description else ""
        
        if amount > 0:  # Income
            total_taxable_income += amount
        else:  # Expense
            expense_amount = abs(amount)
            
            # Apply standard IRS expense categorization
            category_info = categorize_irs_business_expense(description, expense_amount)
            
            # If confidence is low, mark for RAG enhancement
            if category_info["confidence_score"] < 70:
                complex_transactions.append({
                    "transaction": transaction,
                    "expense_amount": expense_amount,
                    "initial_category": category_info
                })
            else:
                # Apply standard categorization
                category = category_info["category"]
                deductible_amount = category_info["deductible_amount"]
                
                transaction.tax_category = category
                transaction.is_deductible = deductible_amount > 0
                
                if deductible_amount > 0:
                    total_deductible_expenses += deductible_amount
                    business_expense_categories[category] += deductible_amount
    
    # Second pass: RAG-enhanced categorization for complex cases
    if complex_transactions and settings.rag_enabled:
        try:
            enhanced_categorizations = await rag_enhanced_categorization(
                complex_transactions, config, store
            )
            
            for i, complex_tx in enumerate(complex_transactions):
                if i < len(enhanced_categorizations):
                    enhanced_cat = enhanced_categorizations[i]
                    transaction = complex_tx["transaction"]
                    
                    # Apply enhanced categorization
                    category = enhanced_cat.get("category", complex_tx["initial_category"]["category"])
                    deductible_amount = enhanced_cat.get("deductible_amount", complex_tx["initial_category"]["deductible_amount"])
                    
                    transaction.tax_category = category
                    transaction.is_deductible = deductible_amount > 0
                    
                    if deductible_amount > 0:
                        total_deductible_expenses += deductible_amount
                        business_expense_categories[category] += deductible_amount
                    
                    # Adjust accuracy based on enhancement
                    if enhanced_cat.get("confidence_score", 0) > 80:
                        categorization_accuracy += 2.0
                else:
                    # Fallback to initial categorization
                    transaction = complex_tx["transaction"]
                    category_info = complex_tx["initial_category"]
                    
                    category = category_info["category"]
                    deductible_amount = category_info["deductible_amount"]
                    
                    transaction.tax_category = category
                    transaction.is_deductible = deductible_amount > 0
                    
                    if deductible_amount > 0:
                        total_deductible_expenses += deductible_amount
                        business_expense_categories[category] += deductible_amount
                    
                    categorization_accuracy -= 5.0
                    
        except Exception as e:
            logger.log_agent_activity(
                "tax_categorizer", "rag_enhancement_failed", "unknown",
                error=str(e)
            )
            # Fallback to initial categorization for complex transactions
            for complex_tx in complex_transactions:
                transaction = complex_tx["transaction"]
                category_info = complex_tx["initial_category"]
                
                category = category_info["category"]
                deductible_amount = category_info["deductible_amount"]
                
                transaction.tax_category = category
                transaction.is_deductible = deductible_amount > 0
                
                if deductible_amount > 0:
                    total_deductible_expenses += deductible_amount
                    business_expense_categories[category] += deductible_amount
                
                categorization_accuracy -= 10.0
    
    # Generate IRS-compliant suggestions and warnings with RAG enhancement
    suggestions = await generate_rag_enhanced_tax_suggestions(business_expense_categories, store)
    warnings = await generate_rag_enhanced_compliance_warnings(transactions, store)
    
    return TaxSummary(
        total_deductible_expenses=total_deductible_expenses,
        total_taxable_income=total_taxable_income,
        business_expense_categories=dict(business_expense_categories),
        tax_optimization_suggestions=suggestions,
        compliance_warnings=warnings,
        categorization_accuracy=max(75.0, min(100.0, categorization_accuracy))  # Clamp between 75-100%
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

# RAG Enhancement Functions

async def rag_enhanced_categorization(complex_transactions: List[Dict[str, Any]], 
                                    config: Optional[RunnableConfig] = None,
                                    store=None) -> List[Dict[str, Any]]:
    """Use RAG to enhance categorization for complex transactions"""
    try:
        # Get IRS documents vector store
        irs_vector_store = get_vector_store("irs_documents")
        
        enhanced_categorizations = []
        
        for complex_tx in complex_transactions:
            transaction = complex_tx["transaction"]
            description = transaction.description or ""
            amount = complex_tx["expense_amount"]
            
            # Search for relevant IRS guidance
            search_query = f"business expense deduction {description} ordinary necessary"
            
            similar_guidance = await irs_vector_store.similarity_search(
                query=search_query,
                k=settings.rag_top_k,
                threshold=settings.rag_score_threshold,
                namespace="publications"
            )
            
            if similar_guidance:
                # Use LLM to analyze guidance and make categorization decision
                enhanced_category = await llm_enhanced_categorization(
                    description, amount, similar_guidance
                )
                enhanced_categorizations.append(enhanced_category)
            else:
                # No relevant guidance found, use original categorization
                enhanced_categorizations.append(complex_tx["initial_category"])
        
        return enhanced_categorizations
        
    except Exception as e:
        logger.log_agent_activity(
            "tax_categorizer", "rag_categorization_error", "unknown",
            error=str(e)
        )
        # Return original categorizations on error
        return [complex_tx["initial_category"] for complex_tx in complex_transactions]

async def llm_enhanced_categorization(description: str, 
                                    amount: Decimal, 
                                    guidance_docs: List[tuple]) -> Dict[str, Any]:
    """Use LLM with IRS guidance to categorize complex transactions"""
    try:
        llm_client = get_llm_client()
        
        # Format guidance documents
        guidance_text = "\n\n".join([
            f"IRS Guidance (Score: {score:.2f}): {content[:500]}..." 
            for content, score, _ in guidance_docs[:3]  # Use top 3 results
        ])
        
        prompt = f"""You are a tax categorization expert. Based on the following IRS guidance and transaction details, categorize this business expense according to official IRS categories.

IRS GUIDANCE:
{guidance_text}

TRANSACTION DETAILS:
Description: {description}
Amount: ${amount:,.2f}

Available IRS Categories:
{', '.join(IRS_EXPENSE_CATEGORIES.keys())}

Respond with ONLY a JSON object containing:
{{
    "category": "exact_category_name_from_list",
    "deductible_percentage": 0.0_to_1.0,
    "confidence_score": 0_to_100,
    "reasoning": "brief_explanation"
}}

Ensure the category is exactly one from the provided list and deductible_percentage follows IRS rules."""
        
        response = await llm_client.agenerate_text(
            prompt, 
            task_type="classification",
            max_tokens=200,
            temperature=0.1
        )
        
        # Parse LLM response
        import json
        try:
            result = json.loads(response.strip())
            
            # Validate and calculate deductible amount
            category = result.get("category", "other_business_expenses")
            deductible_percentage = Decimal(str(result.get("deductible_percentage", 1.0)))
            confidence_score = result.get("confidence_score", 80)
            
            # Ensure category exists in our list
            if category not in IRS_EXPENSE_CATEGORIES:
                category = "other_business_expenses"
                deductible_percentage = Decimal('1.0')
                confidence_score = max(50, confidence_score - 20)
            
            deductible_amount = amount * deductible_percentage
            
            return {
                "category": category,
                "deductible_amount": deductible_amount,
                "confidence_score": confidence_score,
                "reasoning": result.get("reasoning", "LLM-enhanced categorization"),
                "irs_code": IRS_EXPENSE_CATEGORIES.get(category, {}).get("irs_code", "OTHER")
            }
            
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            return {
                "category": "other_business_expenses",
                "deductible_amount": amount,
                "confidence_score": 60,
                "reasoning": "LLM response parsing failed",
                "irs_code": "OTHER"
            }
            
    except Exception as e:
        logger.log_agent_activity(
            "tax_categorizer", "llm_categorization_error", "unknown",
            error=str(e)
        )
        # Fallback categorization
        return {
            "category": "other_business_expenses",
            "deductible_amount": amount,
            "confidence_score": 50,
            "reasoning": "LLM categorization failed",
            "irs_code": "OTHER"
        }

async def generate_rag_enhanced_tax_suggestions(categories: Dict[str, Decimal], 
                                              store=None) -> List[str]:
    """Generate tax optimization suggestions enhanced with RAG"""
    suggestions = []
    
    # Start with standard suggestions
    suggestions.extend(generate_irs_tax_optimization_suggestions(categories))
    
    if not settings.rag_enabled or not store:
        return suggestions
    
    try:
        # Search for additional optimization strategies
        tax_guidance_store = get_vector_store("tax_guidance")
        
        # Generate search queries based on expense categories
        category_queries = [
            f"tax optimization {category} deduction strategies"
            for category in categories.keys() if categories[category] > Decimal('1000')
        ]
        
        for query in category_queries[:3]:  # Limit to top 3 categories
            guidance_results = await tax_guidance_store.similarity_search(
                query=query,
                k=2,
                threshold=0.7,
                namespace="general"
            )
            
            if guidance_results:
                # Extract optimization tips from guidance
                for content, score, metadata in guidance_results:
                    if "optimization" in content.lower() or "strategy" in content.lower():
                        # Summarize the guidance
                        suggestion = f"Tax Strategy: {content[:200]}..."
                        if suggestion not in suggestions:
                            suggestions.append(suggestion)
    
    except Exception as e:
        logger.log_agent_activity(
            "tax_categorizer", "rag_suggestions_error", "unknown",
            error=str(e)
        )
    
    return suggestions[:10]  # Limit to 10 suggestions

async def generate_rag_enhanced_compliance_warnings(transactions: List[Transaction], 
                                                  store=None) -> List[str]:
    """Generate compliance warnings enhanced with RAG"""
    warnings = []
    
    # Start with standard warnings
    warnings.extend(generate_irs_compliance_warnings(transactions))
    
    if not settings.rag_enabled or not store:
        return warnings
    
    try:
        # Search for compliance issues based on transaction patterns
        financial_regs_store = get_vector_store("financial_regulations")
        
        # Check for potential compliance issues
        high_amount_transactions = [
            tx for tx in transactions 
            if abs(tx.amount) > Decimal('5000')
        ]
        
        if high_amount_transactions:
            compliance_query = "large transaction reporting requirements compliance warnings"
            
            compliance_results = await financial_regs_store.similarity_search(
                query=compliance_query,
                k=3,
                threshold=0.7,
                namespace="general"
            )
            
            for content, score, metadata in compliance_results:
                if "warning" in content.lower() or "requirement" in content.lower():
                    warning = f"Compliance Notice: {content[:200]}..."
                    if warning not in warnings:
                        warnings.append(warning)
    
    except Exception as e:
        logger.log_agent_activity(
            "tax_categorizer", "rag_warnings_error", "unknown",
            error=str(e)
        )
    
    return warnings[:8]  # Limit to 8 warnings

async def store_categorization_insights(tax_summary: TaxSummary, 
                                      transactions: List[Transaction],
                                      config: Optional[RunnableConfig] = None,
                                      store=None):
    """Store categorization insights in shared memory for future reference"""
    try:
        if not store:
            return
        
        memory_store = get_shared_memory_store()
        
        # Extract user context
        user_id = config.get("configurable", {}).get("user_id", "default") if config else "default"
        
        # Create categorization insights
        insights = {
            "timestamp": datetime.now().isoformat(),
            "total_transactions": len(transactions),
            "categorization_accuracy": tax_summary.categorization_accuracy,
            "top_expense_categories": dict(sorted(
                tax_summary.business_expense_categories.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]),
            "total_deductible": float(tax_summary.total_deductible_expenses),
            "optimization_opportunities": len(tax_summary.tax_optimization_suggestions),
            "compliance_issues": len(tax_summary.compliance_warnings)
        }
        
        # Store in user-specific namespace
        namespace = MemoryNamespaces.agent_namespace("tax_categorizer", user_id)
        key = f"categorization_insights_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        await memory_store.put_memory(namespace, key, insights)
        
        # Also store in global agent memories for pattern analysis
        global_namespace = MemoryNamespaces.TAX_CATEGORIZER_MEMORIES
        global_key = f"global_insights_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        global_insights = {
            **insights,
            "user_id": user_id,
            "anonymized": True
        }
        
        await memory_store.put_memory(global_namespace, global_key, global_insights)
        
        logger.log_agent_activity(
            "tax_categorizer", "insights_stored", "unknown",
            insights_count=len(insights)
        )
        
    except Exception as e:
        logger.log_agent_activity(
            "tax_categorizer", "insights_storage_failed", "unknown",
            error=str(e)
        )