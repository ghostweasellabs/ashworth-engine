#!/usr/bin/env python3
"""
Debug script to test the extraction logic step by step.
"""

import pandas as pd
from pathlib import Path
from src.utils.file_processors import ExcelProcessor

def debug_extraction():
    """Debug the extraction process step by step."""
    processor = ExcelProcessor()
    
    # Test with overtime.xlsx
    file_path = Path("data/private/overtime.xlsx")
    if not file_path.exists():
        print("File not found")
        return
    
    print("=== DEBUGGING EXTRACTION ===")
    
    # Read the Excel file
    df = pd.read_excel(file_path, sheet_name="2024 TRANSACTIONS", header=None)
    print(f"DataFrame shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Test extraction on a few rows that we know have data
    test_rows = [2, 3, 4, 5]  # Based on our earlier analysis
    
    for row_idx in test_rows:
        if row_idx >= len(df):
            continue
            
        row = df.iloc[row_idx]
        print(f"\n--- Testing Row {row_idx} ---")
        print(f"Raw row data: {dict(row)}")
        
        # Test each extraction method
        date_val, date_issues = processor._extract_date(row)
        print(f"Date extraction: {date_val} (issues: {date_issues})")
        
        amount_val, amount_issues = processor._extract_amount(row)
        print(f"Amount extraction: {amount_val} (issues: {amount_issues})")
        
        desc_val, desc_issues = processor._extract_description(row)
        print(f"Description extraction: '{desc_val}' (issues: {desc_issues})")
        
        # Test full transaction extraction
        try:
            transaction = processor._extract_transaction_from_row(row, row_idx, "test", "debug")
            if transaction:
                print(f"✓ Transaction created:")
                print(f"  ID: {transaction.id}")
                print(f"  Date: {transaction.date}")
                print(f"  Amount: {transaction.amount}")
                print(f"  Description: {transaction.description}")
                print(f"  Quality Score: {transaction.data_quality_score}")
                print(f"  Issues: {transaction.data_issues}")
            else:
                print("✗ No transaction created")
        except Exception as e:
            print(f"✗ Error creating transaction: {e}")

if __name__ == "__main__":
    debug_extraction()