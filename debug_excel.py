#!/usr/bin/env python3
"""
Debug script to examine the messy Excel files and improve extraction logic.
"""

import pandas as pd
from pathlib import Path
import sys

def analyze_excel_file(file_path):
    """Analyze an Excel file in detail."""
    print(f"\n{'='*60}")
    print(f"ANALYZING: {file_path}")
    print(f"{'='*60}")
    
    try:
        xl_file = pd.ExcelFile(file_path)
        print(f"Sheets: {xl_file.sheet_names}")
        
        for sheet_name in xl_file.sheet_names:
            print(f"\n--- Sheet: {sheet_name} ---")
            
            # Read with different strategies
            strategies = [
                {'header': None},
                {'header': 0},
                {'header': 1},
                {'header': 2},
            ]
            
            for i, strategy in enumerate(strategies):
                try:
                    df = pd.read_excel(file_path, sheet_name=sheet_name, **strategy)
                    print(f"\nStrategy {i} ({strategy}): Shape {df.shape}")
                    print("Columns:", list(df.columns))
                    
                    # Show first few rows
                    print("First 5 rows:")
                    for idx, row in df.head().iterrows():
                        print(f"  Row {idx}: {dict(row)}")
                    
                    # Look for potential transaction data
                    print("\nLooking for potential transaction patterns...")
                    
                    # Check each row for date/amount patterns
                    transaction_candidates = []
                    for idx, row in df.iterrows():
                        if idx > 20:  # Don't check too many rows
                            break
                        
                        row_analysis = analyze_row_for_transactions(row, idx)
                        if row_analysis['has_potential']:
                            transaction_candidates.append((idx, row_analysis))
                    
                    if transaction_candidates:
                        print(f"Found {len(transaction_candidates)} potential transaction rows:")
                        for row_idx, analysis in transaction_candidates[:5]:  # Show first 5
                            print(f"  Row {row_idx}: {analysis}")
                    else:
                        print("No clear transaction patterns found")
                    
                    break  # Use first successful strategy
                    
                except Exception as e:
                    print(f"Strategy {i} failed: {e}")
                    continue
            
    except Exception as e:
        print(f"Failed to analyze {file_path}: {e}")

def analyze_row_for_transactions(row, row_idx):
    """Analyze a row to see if it contains transaction-like data."""
    analysis = {
        'has_potential': False,
        'date_candidates': [],
        'amount_candidates': [],
        'text_candidates': [],
        'issues': []
    }
    
    import re
    from datetime import datetime
    
    for col_name, value in row.items():
        if pd.isna(value):
            continue
        
        value_str = str(value).strip()
        if not value_str:
            continue
        
        # Check for date patterns
        date_patterns = [
            r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',
            r'\d{4}[/-]\d{1,2}[/-]\d{1,2}',
            r'\d{1,2}/\d{1,2}',  # MM/DD or DD/MM
        ]
        
        for pattern in date_patterns:
            if re.search(pattern, value_str):
                analysis['date_candidates'].append((col_name, value_str))
                break
        
        # Check for amount patterns
        amount_patterns = [
            r'[\$£€]?\d+\.?\d*',
            r'\(\d+\.?\d*\)',  # Negative amounts in parentheses
            r'\d+,\d{3}',      # Thousands separator
        ]
        
        for pattern in amount_patterns:
            if re.search(pattern, value_str) and len(value_str) <= 20:  # Not too long
                analysis['amount_candidates'].append((col_name, value_str))
                break
        
        # Check for description-like text
        if len(value_str) > 10 and ' ' in value_str:
            analysis['text_candidates'].append((col_name, value_str[:50]))
    
    # Determine if this row has potential
    if (len(analysis['date_candidates']) > 0 or 
        len(analysis['amount_candidates']) > 0) and len(analysis['text_candidates']) > 0:
        analysis['has_potential'] = True
    
    return analysis

if __name__ == "__main__":
    # Analyze the sample files
    sample_files = [
        "data/private/overtime.xlsx",
        "data/private/sitterud.xlsx"
    ]
    
    for file_path in sample_files:
        if Path(file_path).exists():
            analyze_excel_file(file_path)
        else:
            print(f"File not found: {file_path}")