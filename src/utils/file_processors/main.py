"""
Main file processor with format detection and validation.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

# Optional imports with fallbacks
try:
    import magic
    HAS_MAGIC = True
except ImportError:
    HAS_MAGIC = False
    logging.warning("python-magic not available, using fallback file type detection")

from .exceptions import FileValidationError, DataExtractionError
from .excel import ExcelProcessor
from .csv import CSVProcessor
from .pdf import PDFProcessor
from .image import ImageProcessor

logger = logging.getLogger(__name__)


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
        mime_type = self._detect_mime_type(file_path)
        
        if mime_type not in self.SUPPORTED_FORMATS:
            raise FileValidationError(f"Unsupported MIME type: {mime_type}")
        
        file_type = self.SUPPORTED_FORMATS[mime_type]
        logger.info(f"Validated file {file_path}: type={file_type}, size={file_size}")
        
        return file_type, file_size
    
    def _detect_mime_type(self, file_path: Path) -> str:
        """Detect MIME type with fallback to extension-based detection."""
        mime_type = None
        
        if HAS_MAGIC:
            try:
                mime_type = magic.from_file(str(file_path), mime=True)
            except Exception as e:
                logger.warning(f"Could not detect MIME type for {file_path}: {e}")
        
        if not mime_type:
            # Fallback to extension-based detection
            ext = file_path.suffix.lower()
            extension_map = {
                '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                '.xls': 'application/vnd.ms-excel',
                '.csv': 'text/csv',
                '.pdf': 'application/pdf',
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.png': 'image/png',
                '.tif': 'image/tiff',
                '.tiff': 'image/tiff',
            }
            
            mime_type = extension_map.get(ext)
            if not mime_type:
                raise FileValidationError(f"Unsupported file type: {ext}")
        
        return mime_type
    
    def process_file(self, file_path: Union[str, Path], source_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Process a file and extract financial transactions.
        
        Args:
            file_path: Path to the file to process
            source_name: Optional name to use as source identifier
            
        Returns:
            List of raw transaction data dictionaries
            
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