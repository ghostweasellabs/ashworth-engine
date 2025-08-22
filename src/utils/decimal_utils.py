"""Decimal utilities for precise financial calculations"""

from decimal import Decimal, ROUND_HALF_UP, getcontext
from typing import Union, Any

# Set decimal precision for financial calculations
getcontext().prec = 28

def to_decimal(value: Any) -> Decimal:
    """Convert various numeric types to Decimal for precise calculations"""
    if isinstance(value, Decimal):
        return value
    elif isinstance(value, (int, float)):
        return Decimal(str(value))
    elif isinstance(value, str):
        # Clean common formatting from string amounts
        clean_value = value.replace(',', '').replace('$', '').replace('€', '').replace('£', '').strip()
        if clean_value.startswith('(') and clean_value.endswith(')'):
            # Handle negative amounts in parentheses
            clean_value = '-' + clean_value[1:-1]
        return Decimal(clean_value)
    else:
        raise ValueError(f"Cannot convert {type(value)} to Decimal")

def round_currency(amount: Union[Decimal, float, str], places: int = 2) -> Decimal:
    """Round amount to specified decimal places using banker's rounding"""
    decimal_amount = to_decimal(amount)
    return decimal_amount.quantize(Decimal(f"0.{'0' * places}"), rounding=ROUND_HALF_UP)

def format_currency(amount: Union[Decimal, float, str], currency: str = "USD") -> str:
    """Format amount as currency string"""
    decimal_amount = round_currency(amount)
    
    currency_symbols = {
        "USD": "$",
        "EUR": "€", 
        "GBP": "£",
        "JPY": "¥",
        "CAD": "C$",
        "AUD": "A$"
    }
    
    symbol = currency_symbols.get(currency, currency + " ")
    
    if currency == "JPY":
        # Japanese Yen has no decimal places
        return f"{symbol}{int(decimal_amount):,}"
    else:
        return f"{symbol}{decimal_amount:,.2f}"

def calculate_percentage(part: Union[Decimal, float, str], 
                        whole: Union[Decimal, float, str]) -> Decimal:
    """Calculate percentage with proper decimal precision"""
    part_decimal = to_decimal(part)
    whole_decimal = to_decimal(whole)
    
    if whole_decimal == 0:
        return Decimal('0')
    
    return round_currency((part_decimal / whole_decimal) * 100)

def safe_divide(numerator: Union[Decimal, float, str], 
                denominator: Union[Decimal, float, str], 
                default: Decimal = Decimal('0')) -> Decimal:
    """Safely divide two numbers, returning default if denominator is zero"""
    num = to_decimal(numerator)
    den = to_decimal(denominator)
    
    if den == 0:
        return default
    
    return num / den