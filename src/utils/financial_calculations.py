from typing import List, Dict
from decimal import Decimal
from src.workflows.state_schemas import Transaction, FinancialMetrics
import pandas as pd
from collections import defaultdict

def calculate_metrics(transactions: List[Transaction]) -> FinancialMetrics:
    """Calculate comprehensive financial metrics from transactions"""
    
    if not transactions:
        return FinancialMetrics(
            total_revenue=Decimal('0'),
            total_expenses=Decimal('0'),
            gross_profit=Decimal('0'),
            gross_margin_pct=0.0,
            expense_by_category={},
            anomalies=[],
            pattern_matches={},
            detected_business_types=[]
        )
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame([{
        'date': t.date,
        'description': t.description,
        'amount': float(t.amount),
        'category': t.category,
        'account': t.account,
        'currency': t.currency
    } for t in transactions])
    
    # Calculate basic metrics
    total_revenue = Decimal('0')
    total_expenses = Decimal('0')
    
    for transaction in transactions:
        amount = transaction.amount
        if amount > 0:
            total_revenue += amount
        else:
            total_expenses += abs(amount)
    
    # Calculate profit and margin
    gross_profit = total_revenue - total_expenses
    gross_margin_pct = float((gross_profit / total_revenue * 100)) if total_revenue > 0 else 0.0
    
    # Calculate expense breakdown by category
    expense_by_category = calculate_expense_categories(transactions)
    
    # Detect patterns and anomalies
    anomalies = detect_anomalies(transactions)
    pattern_matches = detect_patterns(transactions)
    business_types = detect_business_types(transactions)
    
    return FinancialMetrics(
        total_revenue=total_revenue,
        total_expenses=total_expenses,
        gross_profit=gross_profit,
        gross_margin_pct=gross_margin_pct,
        expense_by_category=expense_by_category,
        anomalies=anomalies,
        pattern_matches=pattern_matches,
        detected_business_types=business_types
    )

def calculate_expense_categories(transactions: List[Transaction]) -> Dict[str, Decimal]:
    """Calculate expenses broken down by category"""
    categories = defaultdict(Decimal)
    
    for transaction in transactions:
        if transaction.amount < 0:  # Expense
            category = transaction.category or categorize_transaction(transaction.description)
            categories[category] += abs(transaction.amount)
    
    # If no specific categories, create basic ones
    if not categories:
        categories["uncategorized"] = sum(abs(t.amount) for t in transactions if t.amount < 0)
    
    return dict(categories)

def categorize_transaction(description: str) -> str:
    """Basic transaction categorization based on description"""
    if not description:
        return "uncategorized"
    
    desc_lower = description.lower()
    # TODO: These are not all of the categories, just a few examples - should follow NAICS standards...
    
    # Office and supplies
    if any(keyword in desc_lower for keyword in ['office', 'supply', 'staples', 'depot']):
        return "office_supplies"
    
    # Travel and meals
    if any(keyword in desc_lower for keyword in ['uber', 'lyft', 'taxi', 'hotel', 'airline', 'restaurant']):
        return "travel_meals"
    
    # Software and subscriptions
    if any(keyword in desc_lower for keyword in ['software', 'subscription', 'saas', 'monthly']):
        return "software_subscriptions"
    
    # Marketing and advertising
    if any(keyword in desc_lower for keyword in ['marketing', 'advertising', 'google ads', 'facebook']):
        return "marketing"
    
    # Professional services
    if any(keyword in desc_lower for keyword in ['consulting', 'legal', 'accounting', 'professional']):
        return "professional_services"
    
    # Utilities
    if any(keyword in desc_lower for keyword in ['electric', 'gas', 'water', 'internet', 'phone']):
        return "utilities"
    
    return "other_expenses"

def detect_anomalies(transactions: List[Transaction]) -> List[str]:
    """Detect potential anomalies in transaction data"""
    anomalies = []
    
    if not transactions:
        return anomalies
    
    amounts = [float(abs(t.amount)) for t in transactions]
    
    # Calculate basic statistics
    mean_amount = sum(amounts) / len(amounts)
    max_amount = max(amounts)
    
    # Large transaction anomaly (more than 10x average)
    large_threshold = mean_amount * 10
    large_transactions = [t for t in transactions if float(abs(t.amount)) > large_threshold]
    
    if large_transactions:
        anomalies.append(f"Found {len(large_transactions)} unusually large transactions")
    
    # Duplicate transaction detection
    transaction_signatures = {}
    for t in transactions:
        signature = f"{t.date}_{t.amount}_{t.description[:20]}"
        if signature in transaction_signatures:
            anomalies.append("Potential duplicate transactions detected")
            break
        transaction_signatures[signature] = True
    
    # Weekend transaction anomaly (for business accounts)
    weekend_transactions = [t for t in transactions if is_weekend_date(t.date)]
    if len(weekend_transactions) > len(transactions) * 0.3:  # More than 30% on weekends
        anomalies.append("High volume of weekend transactions detected")
    
    return anomalies

def detect_patterns(transactions: List[Transaction]) -> Dict[str, any]:
    """Detect spending and income patterns"""
    if not transactions:
        return {}
    
    patterns = {}
    
    # Vendor frequency
    vendor_counts = defaultdict(int)
    for t in transactions:
        if t.amount < 0:  # Expenses only
            vendor = extract_vendor_from_description(t.description)
            vendor_counts[vendor] += 1
    
    patterns["vendor_count"] = len(vendor_counts)
    patterns["top_vendor"] = max(vendor_counts.items(), key=lambda x: x[1])[0] if vendor_counts else "Unknown"
    
    # Monthly spending pattern
    monthly_spending = defaultdict(Decimal)
    for t in transactions:
        if t.amount < 0:
            month = t.date[:7]  # YYYY-MM format
            monthly_spending[month] += abs(t.amount)
    
    patterns["monthly_variation"] = len(set(monthly_spending.values())) > 1
    
    # Transaction frequency
    patterns["total_transactions"] = len(transactions)
    patterns["expense_transactions"] = len([t for t in transactions if t.amount < 0])
    patterns["income_transactions"] = len([t for t in transactions if t.amount > 0])
    
    return patterns

def detect_business_types(transactions: List[Transaction]) -> List[str]:
    """Detect likely business types based on transaction patterns"""
    business_types = []
    
    if not transactions:
        return business_types
    
    # Analyze transaction descriptions for business indicators
    descriptions = [t.description.lower() for t in transactions]
    all_text = " ".join(descriptions)
    
    # Consulting/Professional Services
    if any(keyword in all_text for keyword in ['consulting', 'professional', 'advisory', 'legal', 'accounting']):
        business_types.append("consulting")
    
    # Retail/E-commerce
    if any(keyword in all_text for keyword in ['amazon', 'shopify', 'square', 'paypal', 'stripe']):
        business_types.append("retail")
    
    # Technology/Software
    if any(keyword in all_text for keyword in ['software', 'saas', 'hosting', 'domain', 'api']):
        business_types.append("technology")
    
    # Food Service
    if any(keyword in all_text for keyword in ['restaurant', 'food', 'catering', 'delivery']):
        business_types.append("food_service")
    
    # Default to general business if no specific type detected
    if not business_types:
        business_types.append("general_business")
    
    return business_types

def extract_vendor_from_description(description: str) -> str:
    """Extract vendor name from transaction description"""
    if not description:
        return "Unknown"
    
    # Simple vendor extraction - take first meaningful word
    words = description.split()
    if words:
        # Remove common banking prefixes
        first_word = words[0].upper()
        if first_word in ['POS', 'ATM', 'DEBIT', 'CREDIT']:
            return words[1] if len(words) > 1 else first_word
        return first_word[:20]  # Limit length
    
    return "Unknown"

def is_weekend_date(date_str: str) -> bool:
    """Check if a date string represents a weekend"""
    try:
        from datetime import datetime
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        return date_obj.weekday() >= 5  # Saturday = 5, Sunday = 6
    except:
        return False