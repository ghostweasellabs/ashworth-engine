"""
File processing utilities for handling real-world messy financial data.

This module provides robust parsers for CSV, Excel, and PDF files with extensive
error handling and data normalization capabilities.
"""

import io
import logging
import pandas as pd
import re
from decimal import Decimal, InvalidOperation
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import PyPDF2
import chardet

# Optional imports with fallbacks
try:
    import magic
    HAS_MAGIC = True
except ImportError:
    HAS_MAGIC = False
    logging.warning("python-magic not available, using fallback file type detection")

try:
    import pytesseract
    from PIL import Image
    HAS_OCR = True
except ImportError:
    HAS_OCR = False
    logging.warning("OCR libraries not available, image processing disabled")

from ..models.base import Transaction

logger = logging.getLogger(__name__)


class FileValidationError(Exception):
    """Raised when file validation fails."""
    pass


class DataExtractionError(Exception):
    """Raised when data extraction fails."""
    pass


class FileProcessor:
    """Main file processor with format detection and validation."""
    
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
    SUPPORTED_FORMATS = {
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': 'excel',
        'application/vnd.ms-excel': 'excel',
        'text/csv': 'csv',
        'application/pdf': 'pdf',
        'image/jpeg': 'image',
        'image/png': 'image',
        'image/tiff': 'image',
        'text/plain': 'csv',  # Sometimes CSV files are detected as text/plain
    }
    
    def __init__(self):
        self.processors = {
            'excel': ExcelProcessor(),
            'csv': CSVProcessor(),
            'pdf': PDFProcessor(),
            'image': ImageProcessor(),
        }
    
    def validate_file(self, file_path: Union[str, Path]) -> Tuple[str, int]:
        """
        Validate file format and size.
        
        Returns:
            Tuple of (file_type, file_size)
            
        Raises:
            FileValidationError: If file is invalid
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileValidationError(f"File does not exist: {file_path}")
        
        # Check file size
        file_size = file_path.stat().st_size
        if file_size > self.MAX_FILE_SIZE:
            raise FileValidationError(
                f"File too large: {file_size / 1024 / 1024:.1f}MB "
                f"(max {self.MAX_FILE_SIZE / 1024 / 1024}MB)"
            )
        
        if file_size == 0:
            raise FileValidationError("File is empty")
        
        # Detect MIME type
        mime_type = None
        if HAS_MAGIC:
            try:
                mime_type = magic.from_file(str(file_path), mime=True)
            except Exception as e:
                logger.warning(f"Could not detect MIME type for {file_path}: {e}")
        
        if not mime_type:
            # Fallback to extension-based detection
            ext = file_path.suffix.lower()
            if ext in ['.xlsx', '.xls']:
                mime_type = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            elif ext == '.csv':
                mime_type = 'text/csv'
            elif ext == '.pdf':
                mime_type = 'application/pdf'
            elif ext in ['.jpg', '.jpeg']:
                mime_type = 'image/jpeg'
            elif ext == '.png':
                mime_type = 'image/png'
            elif ext in ['.tif', '.tiff']:
                mime_type = 'image/tiff'
            else:
                raise FileValidationError(f"Unsupported file type: {ext}")
        
        if mime_type not in self.SUPPORTED_FORMATS:
            raise FileValidationError(f"Unsupported MIME type: {mime_type}")
        
        file_type = self.SUPPORTED_FORMATS[mime_type]
        logger.info(f"Validated file {file_path}: type={file_type}, size={file_size}")
        
        return file_type, file_size
    
    def process_file(self, file_path: Union[str, Path], source_name: Optional[str] = None) -> List[Transaction]:
        """
        Process a file and extract financial transactions.
        
        Args:
            file_path: Path to the file to process
            source_name: Optional name to use as source identifier
            
        Returns:
            List of Transaction objects
            
        Raises:
            FileValidationError: If file validation fails
            DataExtractionError: If data extraction fails
        """
        file_path = Path(file_path)
        source_name = source_name or file_path.name
        
        # Validate file
        file_type, file_size = self.validate_file(file_path)
        
        # Process file
        processor = self.processors[file_type]
        try:
            transactions = processor.process(file_path, source_name)
            logger.info(f"Successfully processed {file_path}: {len(transactions)} transactions")
            return transactions
        except Exception as e:
            logger.error(f"Failed to process {file_path}: {e}")
            raise DataExtractionError(f"Failed to extract data from {file_path}: {e}")


class BaseFileProcessor:
    """Base class for file processors."""
    
    def process(self, file_path: Path, source_name: str) -> List[Transaction]:
        """Process a file and return transactions."""
        raise NotImplementedError
    
    def _generate_transaction_id(self, row_data: Dict, row_index: int, source_name: str = "unknown") -> str:
        """Generate a unique transaction ID."""
        # Create a simple hash-based ID from row data and index
        data_str = f"{source_name}_{row_index}_{str(row_data)}"
        return f"txn_{hash(data_str) % 1000000:06d}"
    
    def _calculate_data_quality_score(self, issues: List[str], total_fields: int) -> float:
        """Calculate data quality score based on issues found."""
        if total_fields == 0:
            return 0.0
        
        # Weight different types of issues
        issue_weights = {
            'missing_date': 0.3,
            'missing_amount': 0.4,
            'missing_description': 0.2,
            'invalid_date': 0.2,
            'invalid_amount': 0.3,
            'malformed_data': 0.1,
        }
        
        total_penalty = sum(issue_weights.get(issue, 0.1) for issue in issues)
        score = max(0.0, 1.0 - (total_penalty / total_fields))
        return min(1.0, score)


class ExcelProcessor(BaseFileProcessor):
    """Processor for Excel files (.xlsx, .xls)."""
    
    def process(self, file_path: Path, source_name: str) -> List[Transaction]:
        """Process Excel file with robust error handling for messy data."""
        transactions = []
        
        try:
            # Read Excel file with error handling
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
    
    def _process_excel_sheet(self, file_path: Path, sheet_name: str, source_name: str) -> List[Transaction]:
        """Process a single Excel sheet."""
        transactions = []
        
        try:
            # Read with multiple strategies to handle messy data
            df = self._read_excel_with_fallback(file_path, sheet_name)
            
            if df.empty:
                logger.warning(f"Sheet '{sheet_name}' is empty")
                return transactions
            
            logger.info(f"Processing sheet '{sheet_name}' with shape {df.shape}")
            
            # Try to identify header row and data structure
            header_row, data_start_row = self._find_data_structure(df)
            
            if header_row is not None:
                # Re-read with proper header
                df = pd.read_excel(file_path, sheet_name=sheet_name, header=header_row)
                df = df.iloc[data_start_row - header_row:]
            
            # Clean and normalize column names
            df = self._clean_column_names(df)
            
            # Extract transactions from each row (limit to first 1000 rows for performance)
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
            {'header': 0},  # Standard header in first row
            {'header': 1},  # Header in second row
            {'header': 2},  # Header in third row
            {'header': None},  # No header, use default column names
        ]
        
        for strategy in strategies:
            try:
                df = pd.read_excel(file_path, sheet_name=sheet_name, **strategy)
                if not df.empty and df.shape[1] > 1:  # At least 2 columns
                    return df
            except Exception as e:
                logger.debug(f"Strategy {strategy} failed: {e}")
                continue
        
        # Last resort: read as raw data
        return pd.read_excel(file_path, sheet_name=sheet_name, header=None)
    
    def _find_data_structure(self, df: pd.DataFrame) -> Tuple[Optional[int], int]:
        """Find header row and data start row in messy Excel data."""
        # Look for common financial keywords in rows
        financial_keywords = [
            'date', 'amount', 'description', 'transaction', 'account',
            'debit', 'credit', 'balance', 'payment', 'deposit', 'withdrawal'
        ]
        
        header_row = None
        data_start_row = 0
        
        for idx, row in df.iterrows():
            if idx > 10:  # Don't search too far
                break
            
            # Convert row to string and check for keywords
            row_str = ' '.join(str(val).lower() for val in row if pd.notna(val))
            
            keyword_count = sum(1 for keyword in financial_keywords if keyword in row_str)
            
            if keyword_count >= 2:  # Found likely header row
                header_row = idx
                data_start_row = idx + 1
                break
        
        return header_row, data_start_row
    
    def _clean_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and normalize column names."""
        # Create a copy to avoid modifying original
        df = df.copy()
        
        # Clean column names
        new_columns = []
        for col in df.columns:
            if pd.isna(col) or str(col).startswith('Unnamed'):
                # Try to infer column name from first few non-null values
                inferred_name = self._infer_column_name(df[col])
                new_columns.append(inferred_name or f'col_{len(new_columns)}')
            else:
                # Clean existing column name
                clean_name = str(col).strip().lower().replace(' ', '_')
                clean_name = re.sub(r'[^\w_]', '', clean_name)
                new_columns.append(clean_name or f'col_{len(new_columns)}')
        
        df.columns = new_columns
        return df
    
    def _infer_column_name(self, series: pd.Series) -> Optional[str]:
        """Infer column name from data patterns."""
        # Get first few non-null values
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
            r'\(\d+\.?\d*\)',  # Negative amounts in parentheses
        ]
        
        for pattern in amount_patterns:
            if any(re.search(pattern, val) for val in sample_values):
                return 'amount'
        
        # Check for description patterns (longer text)
        if any(len(val) > 10 and ' ' in val for val in sample_values):
            return 'description'
        
        return None
    
    def _extract_transaction_from_row(self, row: pd.Series, row_idx: int, 
                                    source_name: str, sheet_name: str) -> Optional[Transaction]:
        """Extract transaction from a row with extensive error handling."""
        issues = []
        
        # Skip completely empty rows
        if pd.isna(row).all():
            return None
        
        # Skip header rows (look for common header keywords)
        row_str = ' '.join(str(val).lower() for val in row if pd.notna(val))
        header_keywords = ['date', 'amount', 'description', 'acct', 'account', 'transaction', 'yr', 'mo']
        if any(keyword in row_str for keyword in header_keywords) and row_idx < 5:
            logger.debug(f"Skipping header row {row_idx}: {row_str[:100]}")
            return None
        
        # Extract date
        date_val, date_issues = self._extract_date(row)
        issues.extend(date_issues)
        
        # Extract amount
        amount_val, amount_issues = self._extract_amount(row)
        issues.extend(amount_issues)
        
        # Extract description
        description_val, desc_issues = self._extract_description(row)
        issues.extend(desc_issues)
        
        # Skip rows without essential data (need at least date OR amount)
        if date_val is None and amount_val is None:
            return None
        
        # Skip rows that look like metadata or file paths
        if description_val and any(indicator in description_val.lower() for indicator in 
                                 [':\\', 'file', '.xlsx', '.csv', 'download', 'cards 20']):
            return None
        
        # Use fallback values for missing essential data
        if date_val is None:
            date_val = datetime.now()
            issues.append('missing_date')
        
        if amount_val is None:
            amount_val = Decimal('0')
            issues.append('missing_amount')
        
        if not description_val:
            description_val = f"Transaction from {sheet_name} row {row_idx}"
            issues.append('missing_description')
        
        # Calculate data quality score
        quality_score = self._calculate_data_quality_score(issues, len(row))
        
        # Generate transaction ID
        transaction_id = self._generate_transaction_id(row.to_dict(), row_idx, source_name)
        
        return Transaction(
            id=transaction_id,
            date=date_val,
            amount=amount_val,
            description=description_val,
            source_file=f"{source_name}:{sheet_name}",
            data_quality_score=quality_score,
            data_issues=issues
        )
    
    def _extract_date(self, row: pd.Series) -> Tuple[Optional[datetime], List[str]]:
        """Extract date from row with multiple format support."""
        issues = []
        
        # Look for date in likely columns first
        date_columns = [col for col in row.index if 'date' in str(col).lower()]
        
        # Also check for datetime objects directly (pandas parsed dates)
        datetime_columns = [col for col in row.index if isinstance(row[col], datetime)]
        
        # Prioritize datetime objects, then date columns, then all columns
        search_order = datetime_columns + date_columns + list(row.index)
        
        for col in search_order:
            val = row[col]
            if pd.isna(val):
                continue
            
            # If it's already a datetime object, use it
            if isinstance(val, datetime):
                return val, issues
            
            # Try to parse as date
            parsed_date = DateNormalizer.parse_date(str(val))
            if parsed_date:
                return parsed_date, issues
        
        issues.append('missing_date')
        return None, issues
    
    def _extract_amount(self, row: pd.Series) -> Tuple[Optional[Decimal], List[str]]:
        """Extract amount from row with multiple format support."""
        issues = []
        
        # Look for amount in likely columns
        amount_columns = [
            col for col in row.index 
            if any(keyword in str(col).lower() for keyword in ['amount', 'debit', 'credit', 'balance'])
        ]
        
        # Also check for numeric columns that could be amounts
        numeric_columns = []
        for col in row.index:
            val = row[col]
            if isinstance(val, (int, float)) and not pd.isna(val):
                # Skip values that are likely years, months, or small integers
                if isinstance(val, int):
                    if val >= 2000 and val <= 2030:  # Likely a year
                        continue
                    if val >= 1 and val <= 12:  # Likely a month
                        continue
                # Include negative values and reasonable amounts
                if val < 0 or abs(val) >= 1:
                    numeric_columns.append(col)
        
        # Prioritize amount columns, then numeric columns, then all columns
        search_order = amount_columns + numeric_columns + list(row.index)
        
        for col in search_order:
            val = row[col]
            if pd.isna(val):
                continue
            
            # If it's already a number, convert to Decimal
            if isinstance(val, (int, float)):
                # Skip values that are likely years, months, or very small amounts
                if isinstance(val, int):
                    if val >= 2000 and val <= 2030:  # Likely a year
                        continue
                    if val >= 1 and val <= 12 and col not in amount_columns:  # Likely a month (unless in amount column)
                        continue
                if abs(val) < 0.01:  # Skip very small amounts (likely zero or rounding errors)
                    continue
                return Decimal(str(val)), issues
            
            # Try to parse as amount
            parsed_amount = AmountNormalizer.parse_amount(str(val))
            if parsed_amount is not None and abs(parsed_amount) >= Decimal('0.01'):
                return parsed_amount, issues
        
        issues.append('missing_amount')
        return None, issues
    
    def _extract_description(self, row: pd.Series) -> Tuple[str, List[str]]:
        """Extract description from row."""
        issues = []
        
        # Look for description in likely columns
        desc_columns = [
            col for col in row.index 
            if any(keyword in str(col).lower() for keyword in ['description', 'memo', 'note', 'detail', 'download'])
        ]
        
        # Also look for code columns that might contain category info
        code_columns = [
            col for col in row.index 
            if any(keyword in str(col).lower() for keyword in ['code', 'category', 'type'])
        ]
        
        # Collect all potential description parts
        desc_parts = []
        
        # First, try dedicated description columns
        for col in desc_columns:
            val = row[col]
            if pd.notna(val) and str(val).strip() and len(str(val).strip()) > 3:
                desc_parts.append(str(val).strip())
        
        # Then add code/category information
        for col in code_columns:
            val = row[col]
            if pd.notna(val) and str(val).strip() and len(str(val).strip()) > 1:
                desc_parts.append(f"[{str(val).strip()}]")
        
        # If no dedicated columns, look for any text fields
        if not desc_parts:
            for col in row.index:
                val = row[col]
                if pd.notna(val):
                    val_str = str(val).strip()
                    if (len(val_str) > 5 and not val_str.isdigit()):
                        # Skip values that look like account numbers or codes
                        if not re.match(r'^[A-Z0-9]{3,10}$', val_str):
                            desc_parts.append(val_str)
        
        if desc_parts:
            # Clean and combine description parts
            cleaned_parts = []
            for part in desc_parts[:3]:  # Limit to first 3 parts
                cleaned = DataCleaner.clean_description(part)
                if cleaned and len(cleaned) > 2:
                    cleaned_parts.append(cleaned)
            
            if cleaned_parts:
                return ' | '.join(cleaned_parts), issues
        
        issues.append('missing_description')
        return '', issues


class CSVProcessor(BaseFileProcessor):
    """Processor for CSV files."""
    
    def process(self, file_path: Path, source_name: str) -> List[Transaction]:
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
            {'encoding': 'utf-8', 'sep': ',', 'error_bad_lines': False},
            {'encoding': 'latin1', 'sep': ','},
        ]
        
        for strategy in strategies:
            try:
                df = pd.read_csv(file_path, **strategy)
                if not df.empty:
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
    
    def _extract_transaction_from_row(self, row: pd.Series, row_idx: int, source_name: str) -> Optional[Transaction]:
        """Extract transaction from CSV row."""
        # Reuse Excel extraction logic
        excel_processor = ExcelProcessor()
        return excel_processor._extract_transaction_from_row(row, row_idx, source_name, "csv")


class PDFProcessor(BaseFileProcessor):
    """Processor for PDF files."""
    
    def process(self, file_path: Path, source_name: str) -> List[Transaction]:
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
    
    def _parse_transactions_from_text(self, text: str, source_name: str) -> List[Transaction]:
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
    
    def _create_transaction_from_match(self, match, line_idx: int, source_name: str, original_line: str) -> Optional[Transaction]:
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
            
            # Assume remaining text is description
            if group and len(group) > 2:
                description_val = group.strip()
        
        # Validate essential fields
        if not date_val:
            date_val = datetime.now()
            issues.append('missing_date')
        
        if amount_val is None:
            amount_val = Decimal('0')
            issues.append('missing_amount')
        
        if not description_val:
            description_val = original_line[:50]  # Use first 50 chars of line
            issues.append('missing_description')
        
        quality_score = self._calculate_data_quality_score(issues, 3)
        transaction_id = self._generate_transaction_id({'line': original_line}, line_idx, source_name)
        
        return Transaction(
            id=transaction_id,
            date=date_val,
            amount=amount_val,
            description=description_val,
            source_file=source_name,
            data_quality_score=quality_score,
            data_issues=issues
        )


class ImageProcessor(BaseFileProcessor):
    """Processor for image files using OCR."""
    
    def process(self, file_path: Path, source_name: str) -> List[Transaction]:
        """Process image file using OCR to extract text."""
        if not HAS_OCR:
            raise DataExtractionError("OCR libraries not available. Install pillow and pytesseract to process images.")
        
        transactions = []
        
        try:
            # Extract text using OCR
            text = self._extract_text_from_image(file_path)
            
            if not text.strip():
                logger.warning(f"No text extracted from image {file_path}")
                return transactions
            
            # Use PDF processor logic to parse text
            pdf_processor = PDFProcessor()
            transactions = pdf_processor._parse_transactions_from_text(text, source_name)
            
        except Exception as e:
            logger.error(f"Failed to process image file {file_path}: {e}")
            raise DataExtractionError(f"Cannot process image file: {e}")
        
        return transactions
    
    def _extract_text_from_image(self, file_path: Path) -> str:
        """Extract text from image using OCR."""
        if not HAS_OCR:
            raise DataExtractionError("OCR libraries not available")
        
        try:
            # Open and preprocess image
            image = Image.open(file_path)
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Extract text using pytesseract
            text = pytesseract.image_to_string(image, config='--psm 6')
            
            return text
        
        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            raise


class DateNormalizer:
    """Utility class for normalizing dates from various formats."""
    
    DATE_FORMATS = [
        '%m/%d/%Y', '%m/%d/%y',
        '%d/%m/%Y', '%d/%m/%y',
        '%Y-%m-%d', '%y-%m-%d',
        '%m-%d-%Y', '%m-%d-%y',
        '%d-%m-%Y', '%d-%m-%y',
        '%B %d, %Y', '%b %d, %Y',
        '%d %B %Y', '%d %b %Y',
        '%Y%m%d',
    ]
    
    @classmethod
    def parse_date(cls, date_str: str) -> Optional[datetime]:
        """Parse date string with multiple format support."""
        if not date_str or pd.isna(date_str):
            return None
        
        date_str = str(date_str).strip()
        
        # Handle Excel date numbers
        try:
            if date_str.replace('.', '').isdigit():
                excel_date = float(date_str)
                if 1 <= excel_date <= 100000:  # Reasonable Excel date range
                    # Excel epoch starts at 1900-01-01
                    base_date = datetime(1900, 1, 1)
                    return base_date + pd.Timedelta(days=excel_date - 2)  # Excel bug adjustment
        except (ValueError, OverflowError):
            pass
        
        # Try standard formats
        for fmt in cls.DATE_FORMATS:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        
        # Try pandas date parser as last resort
        try:
            return pd.to_datetime(date_str, infer_datetime_format=True)
        except (ValueError, TypeError):
            pass
        
        return None


class AmountNormalizer:
    """Utility class for normalizing monetary amounts from various formats."""
    
    @classmethod
    def parse_amount(cls, amount_str: str) -> Optional[Decimal]:
        """Parse amount string with multiple format support."""
        if not amount_str or pd.isna(amount_str):
            return None
        
        amount_str = str(amount_str).strip()
        
        if not amount_str:
            return None
        
        # Handle parentheses as negative
        is_negative = False
        if amount_str.startswith('(') and amount_str.endswith(')'):
            is_negative = True
            amount_str = amount_str[1:-1]
        
        # Remove currency symbols and spaces, but keep structure
        cleaned = re.sub(r'[^\d.,-]', '', amount_str)
        
        # If the cleaned string is very different from original (e.g., "invalid123" -> "123")
        # then it's probably not a valid amount
        if len(cleaned) > 0 and len(amount_str) > len(cleaned) * 2:
            return None
        
        # Handle different decimal separators
        if ',' in cleaned and '.' in cleaned:
            # Determine which is decimal separator
            last_comma = cleaned.rfind(',')
            last_dot = cleaned.rfind('.')
            
            if last_dot > last_comma:
                # Dot is decimal separator
                cleaned = cleaned.replace(',', '')
            else:
                # Comma is decimal separator
                cleaned = cleaned.replace('.', '').replace(',', '.')
        elif ',' in cleaned:
            # Check if comma is thousands separator or decimal
            parts = cleaned.split(',')
            if len(parts) == 2 and len(parts[1]) <= 2:
                # Likely decimal separator
                cleaned = cleaned.replace(',', '.')
            else:
                # Likely thousands separator
                cleaned = cleaned.replace(',', '')
        
        # Handle multiple dots (thousands separators)
        dot_count = cleaned.count('.')
        if dot_count > 1:
            # Keep only the last dot as decimal separator
            parts = cleaned.split('.')
            cleaned = ''.join(parts[:-1]) + '.' + parts[-1]
        
        try:
            amount = Decimal(cleaned)
            return -amount if is_negative else amount
        except (InvalidOperation, ValueError):
            return None


class DataCleaner:
    """Utility class for cleaning and validating extracted data."""
    
    @staticmethod
    def clean_description(description: str) -> str:
        """Clean and normalize description text."""
        if not description:
            return ""
        
        # Remove excessive whitespace
        cleaned = re.sub(r'\s+', ' ', description.strip())
        
        # Remove common OCR artifacts
        cleaned = re.sub(r'[^\w\s\-.,()&@#]', '', cleaned)
        
        # Limit length
        if len(cleaned) > 200:
            cleaned = cleaned[:197] + "..."
        
        return cleaned
    
    @staticmethod
    def validate_transaction(transaction: Transaction) -> List[str]:
        """Validate transaction and return list of validation issues."""
        issues = []
        
        # Check date reasonableness
        if transaction.date:
            current_year = datetime.now().year
            if transaction.date.year < 1900 or transaction.date.year > current_year + 1:
                issues.append('unreasonable_date')
        
        # Check amount reasonableness
        if transaction.amount:
            if abs(transaction.amount) > Decimal('1000000'):  # $1M limit
                issues.append('unreasonable_amount')
        
        # Check description quality
        if len(transaction.description) < 3:
            issues.append('poor_description')
        
        return issues