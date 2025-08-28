"""
File processing utilities for handling real-world messy financial data.

This module provides a simplified interface to the modular file processing system.
For detailed implementations, see the file_processors package.
"""

# Import from the modular structure
from .file_processors import (
    FileProcessor,
    FileValidationError,
    DataExtractionError,
    BaseFileProcessor,
    ExcelProcessor,
    CSVProcessor,
    PDFProcessor,
    ImageProcessor,
    DateNormalizer,
    AmountNormalizer,
    DataCleaner
)

# Re-export for backward compatibility
__all__ = [
    'FileProcessor',
    'FileValidationError',
    'DataExtractionError',
    'BaseFileProcessor',
    'ExcelProcessor',
    'CSVProcessor',
    'PDFProcessor',
    'ImageProcessor',
    'DateNormalizer',
    'AmountNormalizer',
    'DataCleaner'
]