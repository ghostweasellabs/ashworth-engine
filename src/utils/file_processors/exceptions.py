"""
Custom exceptions for file processing operations.
"""


class FileValidationError(Exception):
    """Raised when file validation fails."""
    pass


class DataExtractionError(Exception):
    """Raised when data extraction fails."""
    pass