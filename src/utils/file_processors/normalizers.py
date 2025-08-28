"""
Data normalization utilities for dates, amounts, and text cleaning.
"""

import re
import logging
from decimal import Decimal, InvalidOperation
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


class DateNormalizer:
    """Utility class for parsing and normalizing dates."""
    
    DATE_FORMATS = [
        '%Y-%m-%d',
        '%m/%d/%Y',
        '%d/%m/%Y',
        '%m-%d-%Y',
        '%d-%m-%Y',
        '%Y/%m/%d',
        '%m/%d/%y',
        '%d/%m/%y',
        '%m-%d-%y',
        '%d-%m-%y',
        '%Y%m%d',
        '%B %d, %Y',
        '%b %d, %Y',
        '%d %B %Y',
        '%d %b %Y',
    ]
    
    @classmethod
    def parse_date(cls, date_str: str) -> Optional[datetime]:
        """Parse date string with multiple format support."""
        if not date_str or not isinstance(date_str, str):
            return None
        
        # Clean the date string
        date_str = date_str.strip()
        
        # Skip obviously non-date values
        if len(date_str) < 4 or date_str.isalpha():
            return None
        
        # Try each format
        for fmt in cls.DATE_FORMATS:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        
        # Try pandas date parsing as fallback
        try:
            import pandas as pd
            result = pd.to_datetime(date_str, errors='coerce')
            if pd.isna(result):
                return None
            return result.to_pydatetime()
        except Exception:
            pass
        
        logger.debug(f"Could not parse date: {date_str}")
        return None


class AmountNormalizer:
    """Utility class for parsing and normalizing monetary amounts."""
    
    @classmethod
    def parse_amount(cls, amount_str: str) -> Optional[Decimal]:
        """Parse amount string with multiple format support."""
        if not amount_str or not isinstance(amount_str, str):
            return None
        
        # Clean the amount string
        amount_str = amount_str.strip()
        
        # Skip obviously non-amount values
        if not amount_str or len(amount_str) > 20:
            return None
        
        # Check if it looks like a valid amount (starts with digit, currency, or parenthesis)
        if not re.match(r'^[\d$£€(]', amount_str):
            return None
        
        # Remove currency symbols and spaces
        cleaned = re.sub(r'[^\d.,()-]', '', amount_str)
        
        if not cleaned or not re.search(r'\d', cleaned):
            return None
        
        # Handle parentheses (negative amounts)
        is_negative = cleaned.startswith('(') and cleaned.endswith(')')
        if is_negative:
            cleaned = cleaned[1:-1]
        
        # Handle comma as thousands separator
        if ',' in cleaned and '.' in cleaned:
            # Format like 1,234.56 or €1.234,56 (European)
            comma_pos = cleaned.rfind(',')
            dot_pos = cleaned.rfind('.')
            if comma_pos > dot_pos:
                # European format: 1.234,56
                cleaned = cleaned.replace('.', '').replace(',', '.')
            else:
                # US format: 1,234.56
                cleaned = cleaned.replace(',', '')
        elif ',' in cleaned and cleaned.count(',') == 1:
            # Could be European format (1234,56) or thousands (1,234)
            parts = cleaned.split(',')
            if len(parts[1]) == 2:  # Likely decimal separator
                cleaned = cleaned.replace(',', '.')
            else:  # Likely thousands separator
                cleaned = cleaned.replace(',', '')
        
        try:
            amount = Decimal(cleaned)
            return -amount if is_negative else amount
        except (InvalidOperation, ValueError):
            logger.debug(f"Could not parse amount: {amount_str}")
            return None


class DataCleaner:
    """Utility class for cleaning and normalizing text data."""
    
    @classmethod
    def clean_description(cls, description: str) -> str:
        """Clean and normalize description text."""
        if not description or not isinstance(description, str):
            return ""
        
        # Remove extra whitespace
        cleaned = re.sub(r'\s+', ' ', description.strip())
        
        # Remove common file artifacts
        cleaned = re.sub(r'[^\w\s\-.,()&]', '', cleaned)
        
        # Truncate if too long
        if len(cleaned) > 200:
            cleaned = cleaned[:197] + '...'
        
        # Capitalize first letter
        if cleaned:
            cleaned = cleaned[0].upper() + cleaned[1:]
        
        return cleaned