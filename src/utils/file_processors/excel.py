"""
Excel file processor with robust handling for messy real-world data.
"""

import logging
import pandas as pd
import re
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

from .base import BaseFileProcessor
from .exceptions import DataExtractionError
from .normalizers import DateNormalizer, AmountNormalizer, DataCleaner

logger = logging.getLogger(__name__)


class ExcelProcessor(BaseFileProcessor):
    """Processor for Excel files (.xlsx, .xls)."""
    
    def process(self, file_path: Path, source_name: str) -> List[Dict[str, Any]]:
        """Process Excel file with robust error handling for messy data."""
        transactions = []
        
        try:
            xl_file = pd.ExcelFile(file_path)
            logger.info(f"Excel file has sheets: {xl_file.sheet_names}")
            
            # Process each sheet
            for sheet_name in xl_file.sheet_names:
                try:
                    sheet_transactions = self._process_excel_sheet(
                        file_path, sheet_name, source_name
                    )
                    transactions.extend(sheet_transactions)
                except Exception as e:
                    logger.warning(f"Failed to process sheet '{sheet_name}': {e}")
                    continue
            
        except Exception as e:
            logger.error(f"Failed to read Excel file {file_path}: {e}")
            raise DataExtractionError(f"Cannot read Excel file: {e}")
        
        return transactions
    
    def _process_excel_sheet(self, file_path: Path, sheet_name: str, source_name: str) -> List[Dict[str, Any]]:
        """Process a single Excel sheet."""
        transactions = []
        
        try:
            df = self._read_excel_with_fallback(file_path, sheet_name)
            
            if df.empty:
                logger.warning(f"Sheet '{sheet_name}' is empty")
                return transactions
            
            logger.info(f"Processing sheet '{sheet_name}' with shape {df.shape}")
            
            # Try to identify header row and data structure
            header_row, data_start_row = self._find_data_structure(df)
            
            if header_row is not None:
                df = pd.read_excel(file_path, sheet_name=sheet_name, header=header_row)
                df = df.iloc[data_start_row - header_row:]
            
            # Clean and normalize column names
            df = self._clean_column_names(df)
            
            # Extract transactions from each row (limit for performance)
            max_rows = min(1000, len(df))
            for idx in range(max_rows):
                try:
                    row = df.iloc[idx]
                    transaction = self._extract_transaction_from_row(
                        row, idx, source_name, sheet_name
                    )
                    if transaction:
                        transactions.append(transaction)
                except Exception as e:
                    logger.warning(f"Failed to process row {idx}: {e}")
                    continue
            
        except Exception as e:
            logger.error(f"Failed to process Excel sheet '{sheet_name}': {e}")
            raise
        
        return transactions
    
    def _read_excel_with_fallback(self, file_path: Path, sheet_name: str) -> pd.DataFrame:
        """Read Excel with multiple fallback strategies for messy data."""
        strategies = [
            {'header': 0},
            {'header': 1},
            {'header': 2},
            {'header': None},
        ]
        
        for strategy in strategies:
            try:
                df = pd.read_excel(file_path, sheet_name=sheet_name, **strategy)
                if not df.empty and df.shape[1] > 1:
                    return df
            except Exception as e:
                logger.debug(f"Strategy {strategy} failed: {e}")
                continue
        
        return pd.read_excel(file_path, sheet_name=sheet_name, header=None)
    
    def _find_data_structure(self, df: pd.DataFrame) -> Tuple[Optional[int], int]:
        """Find header row and data start row in messy Excel data."""
        financial_keywords = [
            'date', 'amount', 'description', 'transaction', 'account',
            'debit', 'credit', 'balance', 'payment', 'deposit', 'withdrawal'
        ]
        
        header_row = None
        data_start_row = 0
        
        for idx, row in df.iterrows():
            if idx > 10:
                break
            
            row_str = ' '.join(str(val).lower() for val in row if pd.notna(val))
            keyword_count = sum(1 for keyword in financial_keywords if keyword in row_str)
            
            if keyword_count >= 2:
                header_row = idx
                data_start_row = idx + 1
                break
        
        return header_row, data_start_row
    
    def _clean_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and normalize column names."""
        df = df.copy()
        new_columns = []
        
        for col in df.columns:
            if pd.isna(col) or str(col).startswith('Unnamed') or str(col).strip() == '':
                # For unnamed or empty columns, use generic names
                new_columns.append(f'col_{len(new_columns)}')
            else:
                # Clean existing column name
                clean_name = str(col).strip().lower().replace(' ', '_')
                clean_name = re.sub(r'[^\w_]', '', clean_name)
                # Remove trailing underscores
                clean_name = clean_name.rstrip('_')
                new_columns.append(clean_name or f'col_{len(new_columns)}')
        
        df.columns = new_columns
        return df
    
    def _infer_column_name(self, series: pd.Series) -> Optional[str]:
        """Infer column name from data patterns."""
        sample_values = series.dropna().head(10).astype(str)
        
        if sample_values.empty:
            return None
        
        # Check for date patterns
        date_patterns = [
            r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',
            r'\d{4}[/-]\d{1,2}[/-]\d{1,2}',
        ]
        
        for pattern in date_patterns:
            if any(re.search(pattern, val) for val in sample_values):
                return 'date'
        
        # Check for amount patterns
        amount_patterns = [
            r'[\$£€]?\d+\.?\d*',
            r'\(\d+\.?\d*\)',
        ]
        
        for pattern in amount_patterns:
            if any(re.search(pattern, val) for val in sample_values):
                return 'amount'
        
        # Check for description patterns
        if any(len(val) > 10 and ' ' in val for val in sample_values):
            return 'description'
        
        return None
    
    def _extract_transaction_from_row(self, row: pd.Series, row_idx: int, 
                                    source_name: str, sheet_name: str) -> Optional[Dict[str, Any]]:
        """Extract transaction from a row with extensive error handling."""
        issues = []
        
        # Skip completely empty rows
        if pd.isna(row).all():
            return None
        
        # Skip header rows
        row_str = ' '.join(str(val).lower() for val in row if pd.notna(val))
        header_keywords = ['date', 'amount', 'description', 'acct', 'account', 'transaction']
        if any(keyword in row_str for keyword in header_keywords) and row_idx < 5:
            return None
        
        # Extract components
        date_val, date_issues = self._extract_date(row)
        issues.extend(date_issues)
        
        amount_val, amount_issues = self._extract_amount(row)
        issues.extend(amount_issues)
        
        description_val, desc_issues = self._extract_description(row)
        issues.extend(desc_issues)
        
        # Skip rows without essential data
        if date_val is None and amount_val is None:
            return None
        
        # Use fallback values
        if date_val is None:
            date_val = datetime.now()
            issues.append('missing_date')
        
        if amount_val is None:
            amount_val = Decimal('0')
            issues.append('missing_amount')
        
        if not description_val:
            description_val = f"Transaction from {sheet_name} row {row_idx}"
            issues.append('missing_description')
        
        quality_score = self._calculate_data_quality_score(issues, len(row))
        transaction_id = self._generate_transaction_id(row.to_dict(), row_idx, source_name)
        
        return {
            "id": transaction_id,
            "date": date_val.isoformat() if date_val else None,
            "amount": str(amount_val) if amount_val else "0.00",
            "description": description_val,
            "source_file": f"{source_name}:{sheet_name}",
            "data_quality_score": quality_score,
            "data_issues": issues,
            "row_index": row_idx,
            "raw_data": row.to_dict()
        }
    
    def _extract_date(self, row: pd.Series) -> Tuple[Optional[datetime], List[str]]:
        """Extract date from row with multiple format support."""
        issues = []
        
        # Look for date columns first
        date_columns = [col for col in row.index if 'date' in str(col).lower()]
        datetime_columns = [col for col in row.index if isinstance(row[col], datetime)]
        
        search_order = datetime_columns + date_columns + list(row.index)
        
        for col in search_order:
            val = row[col]
            if pd.isna(val):
                continue
            
            if isinstance(val, datetime):
                return val, issues
            
            parsed_date = DateNormalizer.parse_date(str(val))
            if parsed_date:
                return parsed_date, issues
        
        issues.append('missing_date')
        return None, issues
    
    def _extract_amount(self, row: pd.Series) -> Tuple[Optional[Decimal], List[str]]:
        """Extract amount from row with multiple format support."""
        issues = []
        
        # Look for amount columns
        amount_columns = [
            col for col in row.index 
            if any(keyword in str(col).lower() for keyword in ['amount', 'debit', 'credit', 'balance'])
        ]
        
        # Check for numeric columns
        numeric_columns = []
        for col in row.index:
            val = row[col]
            if isinstance(val, (int, float)) and not pd.isna(val):
                if isinstance(val, int) and (2000 <= val <= 2030 or 1 <= val <= 12):
                    continue
                if val < 0 or abs(val) >= 1:
                    numeric_columns.append(col)
        
        search_order = amount_columns + numeric_columns + list(row.index)
        
        for col in search_order:
            val = row[col]
            if pd.isna(val):
                continue
            
            if isinstance(val, (int, float)):
                if isinstance(val, int) and (2000 <= val <= 2030 or 1 <= val <= 12):
                    continue
                if abs(val) < 0.01:
                    continue
                return Decimal(str(val)), issues
            
            parsed_amount = AmountNormalizer.parse_amount(str(val))
            if parsed_amount is not None and abs(parsed_amount) >= Decimal('0.01'):
                return parsed_amount, issues
        
        issues.append('missing_amount')
        return None, issues
    
    def _extract_description(self, row: pd.Series) -> Tuple[str, List[str]]:
        """Extract description from row."""
        issues = []
        
        # Look for description columns
        desc_columns = [
            col for col in row.index 
            if any(keyword in str(col).lower() for keyword in ['description', 'memo', 'note', 'detail'])
        ]
        
        desc_parts = []
        
        # Try dedicated description columns
        for col in desc_columns:
            val = row[col]
            if pd.notna(val) and str(val).strip() and len(str(val).strip()) > 3:
                desc_parts.append(str(val).strip())
        
        # If no dedicated columns, look for text fields
        if not desc_parts:
            for col in row.index:
                val = row[col]
                if pd.notna(val):
                    val_str = str(val).strip()
                    if len(val_str) > 5 and not val_str.isdigit():
                        if not re.match(r'^[A-Z0-9]{3,10}$', val_str):
                            desc_parts.append(val_str)
        
        if desc_parts:
            cleaned_parts = []
            for part in desc_parts[:3]:
                cleaned = DataCleaner.clean_description(part)
                if cleaned and len(cleaned) > 2:
                    cleaned_parts.append(cleaned)
            
            if cleaned_parts:
                return ' | '.join(cleaned_parts), issues
        
        issues.append('missing_description')
        return '', issues