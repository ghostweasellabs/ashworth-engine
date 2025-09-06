#!/usr/bin/env python3
"""
Direct test of Dr. Sharpe to see what she finds and what she does with the data
"""

import asyncio
import os
from src.agents.pdf_document_intelligence import PDFDocumentIntelligenceAgent
from src.workflows.state_schemas import WorkflowState

async def test_sharpe_direct():
    print("🔍 Testing Dr. Sharpe directly...")
    
    # Create a mock workflow state
    state = WorkflowState(
        workflow_id="test-direct",
        client_id="test",
        workflow_type="financial_analysis",
        input_files=["uploads/STATEMENT-XXXXXX2223-2024-01-29.pdf"],
        status="running"
    )
    
    # Initialize Dr. Sharpe
    dr_sharpe = PDFDocumentIntelligenceAgent()
    
    # Run Dr. Sharpe
    print("🕵️‍♀️ Dr. Sharpe is analyzing the PDF...")
    result_state = await dr_sharpe.execute(state)
    
    # Check what Dr. Sharpe found
    pdf_intel = result_state.get("pdf_document_intelligence", {})
    
    print("\n=== DR. SHARPE'S FINDINGS ===")
    print(f"Status: {pdf_intel.get('status')}")
    
    metadata = pdf_intel.get("processing_metadata", {})
    total_transactions = metadata.get("total_transactions", 0)
    print(f"Total transactions found: {total_transactions}")
    
    if total_transactions > 0:
        raw_data = pdf_intel.get("raw_data", [])
        print(f"\n📊 TRANSACTIONS ({len(raw_data)}):")
        for i, transaction in enumerate(raw_data[:10]):  # Show first 10
            print(f"  {i+1}. Date: {transaction.get('date')}")
            print(f"     Amount: ${transaction.get('amount', 0)}")
            print(f"     Description: {transaction.get('description', 'N/A')[:60]}...")
            print(f"     Type: {transaction.get('transaction_type', 'Unknown')}")
            print()
        
        print(f"\n🎯 WHAT DR. SHARPE DOES WITH THE DATA:")
        print(f"✅ Stores {total_transactions} transactions in state['pdf_document_intelligence']['raw_data']")
        print(f"✅ Sets transaction count in state['pdf_document_intelligence']['processing_metadata']['total_transactions']")
        print(f"✅ Adds quality score: {metadata.get('quality_score', 'N/A')}")
        print(f"✅ Adds forensic analysis: {len(str(metadata.get('forensic_analysis', {})))} chars of analysis")
        
        errors = pdf_intel.get("validation_errors", [])
        if errors:
            print(f"⚠️  Validation errors: {len(errors)}")
    else:
        print("❌ No transactions found")
        errors = pdf_intel.get("validation_errors", [])
        if errors:
            print(f"Errors: {errors}")

if __name__ == "__main__":
    asyncio.run(test_sharpe_direct())
