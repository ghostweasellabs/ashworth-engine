"""
PDF file processor with text extraction and pattern matching.
"""

import logging
import re
import PyPDF2
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import List, Optional, Dict, Any

from .base import BaseFileProcessor
from .exceptions import DataExtractionError
from .normalizers import DateNormalizer, AmountNormalizer

logger = logging.getLogger(__name__)


class PDFProcessor(BaseFileProcessor):
    """Processor for PDF files."""
    
    def process(self, file_path: Path, source_name: str) -> List[Dict[str, Any]]:
        """Process PDF file by extracting text and parsing."""
        transactions = []
        
        try:
            # Extract text from PDF
            text = self._extract_text_from_pdf(file_path)
            
            if not text.strip():
                logger.warning(f"No text extracted from PDF {file_path}")
                return transactions
            
            # Parse transactions from text
            transactions = self._parse_transactions_from_text(text, source_name)
            
        except Exception as e:
            logger.error(f"Failed to process PDF file {file_path}: {e}")
            raise DataExtractionError(f"Cannot process PDF file: {e}")
        
        return transactions
    
    def _extract_text_from_pdf(self, file_path: Path) -> str:
        """Extract text from PDF file."""
        text = ""
        
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        text += f"\n--- Page {page_num + 1} ---\n{page_text}"
                    except Exception as e:
                        logger.warning(f"Failed to extract text from page {page_num + 1}: {e}")
                        continue
        
        except Exception as e:
            logger.error(f"Failed to read PDF file: {e}")
            raise
        
        return text
    
    def _parse_transactions_from_text(self, text: str, source_name: str) -> List[Dict[str, Any]]:
        """Parse transactions from extracted PDF text."""
        transactions = []
        
        # Split text into lines
        lines = text.split('\n')
        
        # Look for transaction patterns
        transaction_patterns = [
            # Date Amount Description pattern
            r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\s+([+-]?\$?[\d,]+\.?\d*)\s+(.+)',
            # Amount Date Description pattern
            r'([+-]?\$?[\d,]+\.?\d*)\s+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\s+(.+)',
        ]
        
        for line_idx, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            for pattern in transaction_patterns:
                match = re.search(pattern, line)
                if match:
                    try:
                        transaction = self._create_transaction_from_match(
                            match, line_idx, source_name, line
                        )
                        if transaction:
                            transactions.append(transaction)
                        break
                    except Exception as e:
                        logger.warning(f"Failed to create transaction from line {line_idx}: {e}")
                        continue
        
        return transactions
    
    def _create_transaction_from_match(self, match, line_idx: int, source_name: str, original_line: str) -> Optional[Dict[str, Any]]:
        """Create transaction from regex match."""
        groups = match.groups()
        issues = []
        
        # Try to identify date, amount, and description from groups
        date_val = None
        amount_val = None
        description_val = ""
        
        for group in groups:
            if not date_val:
                parsed_date = DateNormalizer.parse_date(group)
                if parsed_date:
                    date_val = parsed_date
                    continue
            
            if not amount_val:
                parsed_amount = AmountNormalizer.parse_amount(group)
                if parsed_amount is not None:
                    amount_val = parsed_amount
                    continue
            
            # Remaining group is likely description
            if not description_val and len(group.strip()) > 3:
                description_val = group.strip()
        
        # Use fallbacks if needed
        if not date_val:
            date_val = datetime.now()
            issues.append('missing_date')
        
        if not amount_val:
            amount_val = Decimal('0')
            issues.append('missing_amount')
        
        if not description_val:
            description_val = f"PDF transaction line {line_idx}"
            issues.append('missing_description')
        
        quality_score = self._calculate_data_quality_score(issues, 3)
        transaction_id = self._generate_transaction_id(
            {'line': original_line}, line_idx, source_name
        )
        
        return {
            "id": transaction_id,
            "date": date_val.isoformat() if date_val else None,
            "amount": str(amount_val) if amount_val else "0.00",
            "description": description_val,
            "source_file": f"{source_name}:pdf",
            "data_quality_score": quality_score,
            "data_issues": issues,
            "raw_data": {"line": original_line}
        }