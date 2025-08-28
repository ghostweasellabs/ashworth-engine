#!/usr/bin/env python3
"""
Simple test to extract transactions from first 100 rows.
"""

import pandas as pd
from pathlib import Path
from decimal import Decimal
from datetime import datetime
from src.models.base import Transaction

def simple_extraction():
    """Simple extraction test."""
    file_path = Path("data/private/overtime.xlsx")
    if not file_path.exists():
        print("File not found")
        return
    
    # Read just first 100 rows
    df = pd.read_excel(file_path, sheet_name="2024 TRANSACTIONS", header=None, nrows=100)
    print(f"DataFrame shape: {df.shape}")
    
    transactions = []
    
    # Process rows 2-99 (skip headers)
    for idx in range(2, min(100, len(df))):
        row = df.iloc[idx]
        
        # Skip completely empty rows
        if pd.isna(row).all():
            continue
        
        # Extract data directly by column index based on our analysis
        try:
            # Column 4 = date, Column 7 = amount, Columns 9,10,11 = description parts
            date_val = row[4] if pd.notna(row[4]) else None
            amount_val = row[7] if pd.notna(row[7]) else None
            
            # Build description from multiple columns
            desc_parts = []
            for col_idx in [9, 10, 11]:
                if pd.notna(row[col_idx]) and str(row[col_idx]).strip():
                    desc_parts.append(str(row[col_idx]).strip())
            
            description = " | ".join(desc_parts) if desc_parts else f"Transaction row {idx}"
            
            # Validate we have essential data
            if date_val is None or amount_val is None:
                continue
            
            # Convert types
            if not isinstance(date_val, datetime):
                continue  # Skip if not already a datetime
            
            if not isinstance(amount_val, (int, float)):
                continue  # Skip if not a number
            
            amount_decimal = Decimal(str(amount_val))
            
            # Create transaction
            transaction = Transaction(
                id=f"txn_{idx:06d}",
                date=date_val,
                amount=amount_decimal,
                description=description,
                source_file="overtime.xlsx:2024 TRANSACTIONS",
                data_quality_score=1.0,
                data_issues=[]
            )
            
            transactions.append(transaction)
            
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            continue
    
    print(f"Successfully extracted {len(transactions)} transactions")
    
    # Show first few
    for i, txn in enumerate(transactions[:5]):
        print(f"\nTransaction {i+1}:")
        print(f"  Date: {txn.date}")
        print(f"  Amount: {txn.amount}")
        print(f"  Description: {txn.description}")

if __name__ == "__main__":
    simple_extraction()