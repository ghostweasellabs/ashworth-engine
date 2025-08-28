#!/usr/bin/env python3
"""
Debug the full file processing workflow.
"""

from pathlib import Path
from src.utils.file_processors import ExcelProcessor

def debug_full_process():
    """Debug the full processing workflow."""
    processor = ExcelProcessor()
    
    file_path = Path("data/private/overtime.xlsx")
    if not file_path.exists():
        print("File not found")
        return
    
    print("=== DEBUGGING FULL PROCESS ===")
    
    try:
        transactions = processor.process(file_path, "test_file")
        print(f"Total transactions found: {len(transactions)}")
        
        # Show first few transactions
        for i, txn in enumerate(transactions[:5]):
            print(f"\nTransaction {i+1}:")
            print(f"  ID: {txn.id}")
            print(f"  Date: {txn.date}")
            print(f"  Amount: {txn.amount}")
            print(f"  Description: {txn.description[:100]}...")
            print(f"  Quality Score: {txn.data_quality_score}")
            print(f"  Issues: {txn.data_issues}")
            
    except Exception as e:
        print(f"Error in full process: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_full_process()