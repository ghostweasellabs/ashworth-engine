"""
File processing utilities for handling real-world messy financial data.

This package provides robust parsers for CSV, Excel, and PDF files with extensive
error handling and data normalization capabilities.
"""

from .main import FileProcessor
from .exceptions import FileValidationError, DataExtractionError
from .base import BaseFileProcessor
from .excel import ExcelProcessor
from .csv import CSVProcessor
from .pdf import PDFProcessor
from .image import ImageProcessor
from .normalizers import DateNormalizer, AmountNormalizer, DataCleaner

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