"""
CSV file processor with encoding detection and error handling.
"""

import logging
import pandas as pd
import re
import chardet
from pathlib import Path
from typing import List, Optional, Dict, Any

from .base import BaseFileProcessor
from .exceptions import DataExtractionError
from .excel import ExcelProcessor

logger = logging.getLogger(__name__)


class CSVProcessor(BaseFileProcessor):
    """Processor for CSV files."""
    
    def process(self, file_path: Path, source_name: str) -> List[Dict[str, Any]]:
        """Process CSV file with encoding detection and error handling."""
        transactions = []
        
        # Detect encoding
        encoding = self._detect_encoding(file_path)
        
        try:
            # Try multiple CSV reading strategies
            df = self._read_csv_with_fallback(file_path, encoding)
            
            if df.empty:
                logger.warning(f"CSV file {file_path} is empty")
                return transactions
            
            logger.info(f"Processing CSV with shape {df.shape}")
            
            # Clean column names
            df = self._clean_column_names(df)
            
            # Extract transactions
            for idx, row in df.iterrows():
                try:
                    transaction = self._extract_transaction_from_row(row, idx, source_name)
                    if transaction:
                        transactions.append(transaction)
                except Exception as e:
                    logger.warning(f"Failed to process CSV row {idx}: {e}")
                    continue
            
        except Exception as e:
            logger.error(f"Failed to process CSV file {file_path}: {e}")
            raise DataExtractionError(f"Cannot process CSV file: {e}")
        
        return transactions
    
    def _detect_encoding(self, file_path: Path) -> str:
        """Detect file encoding using chardet."""
        try:
            with open(file_path, 'rb') as f:
                raw_data = f.read(10000)  # Read first 10KB
                result = chardet.detect(raw_data)
                encoding = result['encoding']
                confidence = result['confidence']
                
                logger.info(f"Detected encoding: {encoding} (confidence: {confidence:.2f})")
                
                # Use UTF-8 as fallback for low confidence
                if confidence < 0.7:
                    encoding = 'utf-8'
                
                return encoding or 'utf-8'
        except Exception as e:
            logger.warning(f"Encoding detection failed: {e}, using utf-8")
            return 'utf-8'
    
    def _read_csv_with_fallback(self, file_path: Path, encoding: str) -> pd.DataFrame:
        """Read CSV with multiple fallback strategies."""
        strategies = [
            {'encoding': encoding, 'sep': ','},
            {'encoding': encoding, 'sep': ';'},
            {'encoding': encoding, 'sep': '\t'},
            {'encoding': 'utf-8', 'sep': ',', 'on_bad_lines': 'skip'},
            {'encoding': 'utf-8', 'sep': ';', 'on_bad_lines': 'skip'},
            {'encoding': 'latin1', 'sep': ','},
        ]
        
        for strategy in strategies:
            try:
                df = pd.read_csv(file_path, **strategy)
                if not df.empty and df.shape[1] > 1:  # Must have multiple columns
                    return df
            except Exception as e:
                logger.debug(f"CSV strategy {strategy} failed: {e}")
                continue
        
        raise DataExtractionError("Could not read CSV file with any strategy")
    
    def _clean_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean CSV column names."""
        df = df.copy()
        new_columns = []
        
        for col in df.columns:
            clean_name = str(col).strip().lower().replace(' ', '_')
            clean_name = re.sub(r'[^\w_]', '', clean_name)
            new_columns.append(clean_name or f'col_{len(new_columns)}')
        
        df.columns = new_columns
        return df
    
    def _extract_transaction_from_row(self, row: pd.Series, row_idx: int, source_name: str) -> Optional[Dict[str, Any]]:
        """Extract transaction from CSV row."""
        # Reuse Excel extraction logic
        excel_processor = ExcelProcessor()
        return excel_processor._extract_transaction_from_row(row, row_idx, source_name, "csv")