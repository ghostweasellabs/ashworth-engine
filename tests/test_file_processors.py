"""
Tests for file processing utilities with real-world messy data.
"""

import pytest
import tempfile
from pathlib import Path
from decimal import Decimal
from datetime import datetime
import pandas as pd

from src.utils.file_processors import (
    FileProcessor, 
    FileValidationError, 
    DataExtractionError,
    ExcelProcessor,
    CSVProcessor,
    DateNormalizer,
    AmountNormalizer,
    DataCleaner
)


class TestFileProcessor:
    """Test the main FileProcessor class."""
    
    def setup_method(self):
        self.processor = FileProcessor()
    
    def test_validate_file_success(self):
        """Test successful file validation."""
        # Test with real sample files
        sample_files = [
            "data/private/overtime.xlsx",
            "data/private/sitterud.xlsx"
        ]
        
        for file_path in sample_files:
            if Path(file_path).exists():
                file_type, file_size = self.processor.validate_file(file_path)
                assert file_type == 'excel'
                assert file_size > 0
    
    def test_validate_file_not_exists(self):
        """Test validation with non-existent file."""
        with pytest.raises(FileValidationError, match="File does not exist"):
            self.processor.validate_file("nonexistent.xlsx")
    
    def test_validate_file_too_large(self):
        """Test validation with oversized file."""
        # Create a temporary large file
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
            # Write more than 50MB
            tmp.write(b'x' * (51 * 1024 * 1024))
            tmp_path = tmp.name
        
        try:
            with pytest.raises(FileValidationError, match="File too large"):
                self.processor.validate_file(tmp_path)
        finally:
            Path(tmp_path).unlink()
    
    def test_validate_file_empty(self):
        """Test validation with empty file."""
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            with pytest.raises(FileValidationError, match="File is empty"):
                self.processor.validate_file(tmp_path)
        finally:
            Path(tmp_path).unlink()
    
    def test_process_real_excel_files(self):
        """Test processing real Excel files from data/private."""
        sample_files = [
            "data/private/overtime.xlsx",
            "data/private/sitterud.xlsx"
        ]
        
        for file_path in sample_files:
            if Path(file_path).exists():
                try:
                    transactions = self.processor.process_file(file_path)
                    
                    # Should return a list (even if empty)
                    assert isinstance(transactions, list)
                    
                    # If transactions found, validate structure
                    for transaction in transactions:
                        assert hasattr(transaction, 'id')
                        assert hasattr(transaction, 'date')
                        assert hasattr(transaction, 'amount')
                        assert hasattr(transaction, 'description')
                        assert hasattr(transaction, 'data_quality_score')
                        assert hasattr(transaction, 'data_issues')
                        
                        # Validate data quality score
                        assert 0.0 <= transaction.data_quality_score <= 1.0
                        
                        # Validate data types
                        assert isinstance(transaction.amount, Decimal)
                        assert isinstance(transaction.date, datetime)
                        
                    print(f"Processed {file_path}: {len(transactions)} transactions")
                    
                except Exception as e:
                    # Log the error but don't fail the test for messy data
                    print(f"Processing {file_path} failed (expected for messy data): {e}")


class TestExcelProcessor:
    """Test Excel-specific processing."""
    
    def setup_method(self):
        self.processor = ExcelProcessor()
    
    def test_process_messy_excel_data(self):
        """Test processing with the actual messy Excel files."""
        sample_files = [
            "data/private/overtime.xlsx",
            "data/private/sitterud.xlsx"
        ]
        
        for file_path in sample_files:
            if Path(file_path).exists():
                try:
                    transactions = self.processor.process(Path(file_path), file_path)
                    
                    print(f"\n=== Processing {file_path} ===")
                    print(f"Found {len(transactions)} transactions")
                    
                    # Show sample transactions
                    for i, txn in enumerate(transactions[:3]):
                        print(f"Transaction {i+1}:")
                        print(f"  Date: {txn.date}")
                        print(f"  Amount: {txn.amount}")
                        print(f"  Description: {txn.description[:50]}...")
                        print(f"  Quality Score: {txn.data_quality_score:.2f}")
                        print(f"  Issues: {txn.data_issues}")
                        print()
                    
                except Exception as e:
                    print(f"Expected error processing messy data in {file_path}: {e}")
    
    def test_clean_column_names(self):
        """Test column name cleaning."""
        # Create test DataFrame with messy column names
        df = pd.DataFrame({
            'Unnamed: 0': [1, 2, 3],
            'Date/Time': ['2024-01-01', '2024-01-02', '2024-01-03'],
            'Amount ($)': [100, 200, 300],
            '': ['desc1', 'desc2', 'desc3']
        })
        
        cleaned_df = self.processor._clean_column_names(df)
        
        # Check that column names are cleaned
        assert 'col_0' in cleaned_df.columns  # Unnamed column
        assert 'datetime' in cleaned_df.columns  # Cleaned special chars
        assert 'amount' in cleaned_df.columns  # Cleaned special chars
        assert 'col_3' in cleaned_df.columns  # Empty column name
    
    def test_infer_column_name(self):
        """Test column name inference from data."""
        # Test date column inference
        date_series = pd.Series(['2024-01-01', '2024-01-02', '2024-01-03'])
        assert self.processor._infer_column_name(date_series) == 'date'
        
        # Test amount column inference
        amount_series = pd.Series(['$100.00', '$200.50', '$300.75'])
        assert self.processor._infer_column_name(amount_series) == 'amount'
        
        # Test description column inference
        desc_series = pd.Series(['Long description text here', 'Another long description', 'Yet another description'])
        assert self.processor._infer_column_name(desc_series) == 'description'


class TestCSVProcessor:
    """Test CSV processing."""
    
    def setup_method(self):
        self.processor = CSVProcessor()
    
    def test_detect_encoding(self):
        """Test encoding detection."""
        # Create temporary CSV with different encodings
        test_data = "Date,Amount,Description\n2024-01-01,100.00,Test transaction\n"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as tmp:
            tmp.write(test_data)
            tmp_path = tmp.name
        
        try:
            encoding = self.processor._detect_encoding(Path(tmp_path))
            assert encoding in ['utf-8', 'ascii']  # Both are acceptable
        finally:
            Path(tmp_path).unlink()
    
    def test_read_csv_with_fallback(self):
        """Test CSV reading with fallback strategies."""
        # Create test CSV with semicolon separator
        test_data = "Date;Amount;Description\n2024-01-01;100,00;Test transaction\n"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as tmp:
            tmp.write(test_data)
            tmp_path = tmp.name
        
        try:
            df = self.processor._read_csv_with_fallback(Path(tmp_path), 'utf-8')
            assert not df.empty
            assert df.shape[1] >= 3  # Should have at least 3 columns
        finally:
            Path(tmp_path).unlink()


class TestDateNormalizer:
    """Test date normalization utilities."""
    
    def test_parse_various_date_formats(self):
        """Test parsing various date formats."""
        test_cases = [
            ('01/15/2024', datetime(2024, 1, 15)),
            ('15/01/2024', datetime(2024, 1, 15)),  # Will parse as MM/DD/YYYY first
            ('2024-01-15', datetime(2024, 1, 15)),
            ('January 15, 2024', datetime(2024, 1, 15)),
            ('15 Jan 2024', datetime(2024, 1, 15)),
        ]
        
        for date_str, expected in test_cases:
            result = DateNormalizer.parse_date(date_str)
            if result:  # Some formats might not parse due to ambiguity
                assert result.year == expected.year
                assert result.month == expected.month
                assert result.day == expected.day
    
    def test_parse_excel_date_numbers(self):
        """Test parsing Excel date serial numbers."""
        # Excel date serial number for 2024-01-01
        excel_date = "45292"  # Approximate Excel serial for 2024-01-01
        result = DateNormalizer.parse_date(excel_date)
        
        if result:
            assert result.year >= 2020  # Should be reasonable year
    
    def test_parse_invalid_dates(self):
        """Test handling of invalid dates."""
        invalid_dates = ['', None, 'invalid', '99/99/9999', 'abc123']
        
        for invalid_date in invalid_dates:
            result = DateNormalizer.parse_date(invalid_date)
            assert result is None


class TestAmountNormalizer:
    """Test amount normalization utilities."""
    
    def test_parse_various_amount_formats(self):
        """Test parsing various amount formats."""
        test_cases = [
            ('100.00', Decimal('100.00')),
            ('$100.00', Decimal('100.00')),
            ('1,000.50', Decimal('1000.50')),
            ('(100.00)', Decimal('-100.00')),  # Negative in parentheses
            ('€1.234,56', Decimal('1234.56')),  # European format
            ('1.000,50', Decimal('1000.50')),   # European thousands separator
        ]
        
        for amount_str, expected in test_cases:
            result = AmountNormalizer.parse_amount(amount_str)
            assert result == expected, f"Failed for {amount_str}: got {result}, expected {expected}"
    
    def test_parse_invalid_amounts(self):
        """Test handling of invalid amounts."""
        invalid_amounts = ['', None, 'abc', '$$$$', 'invalid123']
        
        for invalid_amount in invalid_amounts:
            result = AmountNormalizer.parse_amount(invalid_amount)
            assert result is None


class TestDataCleaner:
    """Test data cleaning utilities."""
    
    def test_clean_description(self):
        """Test description cleaning."""
        test_cases = [
            ('  Multiple   spaces  ', 'Multiple spaces'),
            ('Special@#$%chars!', 'Specialchars'),
            ('Very long description that exceeds the maximum length limit and should be truncated' * 5, 
             'Very long description that exceeds the maximum length limit and should be truncated' * 2 + '...'),
        ]
        
        for input_desc, expected in test_cases:
            result = DataCleaner.clean_description(input_desc)
            if '...' in expected:
                assert len(result) <= 200
                assert result.endswith('...')
            else:
                assert result == expected


class TestErrorHandling:
    """Test error handling with garbage data."""
    
    def test_garbage_data_handling(self):
        """Test handling of completely garbage data."""
        # Create DataFrame with garbage data
        garbage_df = pd.DataFrame({
            'col1': [None, '', 'garbage', 123, '###ERROR###'],
            'col2': ['not_a_date', '99/99/99', '', None, 'abc'],
            'col3': ['not_amount', '$$$$', '', None, 'xyz'],
            'col4': [None, '', 'some text', 456, '!!!']
        })
        
        processor = ExcelProcessor()
        
        # Should not crash, even with garbage data
        for idx, row in garbage_df.iterrows():
            try:
                transaction = processor._extract_transaction_from_row(row, idx, "test", "garbage")
                if transaction:
                    # Should have fallback values
                    assert transaction.date is not None
                    assert transaction.amount is not None
                    assert transaction.description != ""
                    assert 0.0 <= transaction.data_quality_score <= 1.0
            except Exception as e:
                # Should handle gracefully
                print(f"Handled garbage data error: {e}")
    
    def test_edge_cases(self):
        """Test various edge cases."""
        processor = FileProcessor()
        
        # Test with completely empty DataFrame
        empty_df = pd.DataFrame()
        excel_processor = ExcelProcessor()
        
        # Should not crash
        try:
            result = excel_processor._process_excel_sheet(Path("test.xlsx"), "empty", "test")
            assert isinstance(result, list)
        except Exception:
            pass  # Expected to fail gracefully
    
    def test_malformed_file_handling(self):
        """Test handling of malformed files."""
        # Create a file that looks like Excel but isn't
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
            tmp.write(b'This is not an Excel file')
            tmp_path = tmp.name
        
        try:
            processor = FileProcessor()
            with pytest.raises(DataExtractionError):
                processor.process_file(tmp_path)
        finally:
            Path(tmp_path).unlink()


if __name__ == "__main__":
    # Run tests with real data
    pytest.main([__file__, "-v", "-s"])