#!/usr/bin/env python3
"""
Demo script to test the Categorizer Agent functionality.
"""

import asyncio
from datetime import datetime
from decimal import Decimal

from src.agents.categorizer import CategorizerAgent
from src.models.base import Transaction
from src.workflows.state_schemas import WorkflowState, AnalysisState, AgentStatus


async def main():
    """Demonstrate the Categorizer Agent functionality."""
    print("🏛️  Ashworth Engine - Categorizer Agent Demo")
    print("=" * 50)
    
    # Create sample transactions
    sample_transactions = [
        Transaction(
            id="tx1",
            date=datetime(2024, 1, 15),
            amount=Decimal("89.47"),
            description="STAPLES OFFICE SUPPLIES",
            counterparty="Staples Inc",
            source_file="demo.csv",
            data_quality_score=0.95,
            data_issues=[]
        ),
        Transaction(
            id="tx2",
            date=datetime(2024, 1, 18),
            amount=Decimal("127.83"),
            description="RESTAURANT MEETING CLIENT JONES",
            counterparty="The Capital Grille",
            source_file="demo.csv",
            data_quality_score=0.9,
            data_issues=[]
        ),
        Transaction(
            id="tx3",
            date=datetime(2024, 1, 22),
            amount=Decimal("2847.99"),
            description="DELL LAPTOP DEVELOPMENT TEAM",
            counterparty="Dell Technologies",
            source_file="demo.csv",
            data_quality_score=0.92,
            data_issues=[]
        ),
        Transaction(
            id="tx4",
            date=datetime(2024, 1, 25),
            amount=Decimal("15000.00"),
            description="EQUIPMENT PURCHASE MANUFACTURING",
            counterparty="Industrial Equipment Corp",
            source_file="demo.csv",
            data_quality_score=0.85,
            data_issues=["Large round amount"]
        ),
        Transaction(
            id="tx5",
            date=datetime(2024, 1, 28),
            amount=Decimal("234.56"),
            description="ELECTRIC BILL OFFICE SPACE",
            counterparty="ConEd",
            source_file="demo.csv",
            data_quality_score=0.94,
            data_issues=[]
        )
    ]
    
    print(f"📊 Processing {len(sample_transactions)} sample transactions...")
    print()
    
    # Create workflow state
    workflow_state = WorkflowState(
        workflow_id="demo-workflow",
        analysis=AnalysisState(
            transactions=[tx.model_dump() for tx in sample_transactions],
            status=AgentStatus.COMPLETED
        )
    )
    
    # Initialize categorizer agent
    categorizer = CategorizerAgent()
    print(f"🤖 Agent: {categorizer.personality.name}")
    print(f"📋 Title: {categorizer.personality.title}")
    print()
    
    # Execute categorization
    print("🔄 Executing tax categorization...")
    result_state = await categorizer.execute(workflow_state)
    
    # Display results
    analysis = result_state["analysis"]
    tax_implications = analysis["tax_implications"]
    
    print("✅ Categorization completed!")
    print()
    
    # Show categorized transactions
    print("📋 CATEGORIZED TRANSACTIONS:")
    print("-" * 50)
    for tx_data in analysis["transactions"]:
        print(f"• {tx_data['description'][:40]:<40} → {tx_data['tax_category']}")
    print()
    
    # Show tax metrics
    print("💰 TAX METRICS:")
    print("-" * 50)
    print(f"Total Deductible: ${tax_implications['total_deductible']}")
    print(f"Estimated Tax Savings: ${tax_implications['tax_savings_estimate']}")
    print()
    
    # Show optimization opportunities
    print("🎯 OPTIMIZATION OPPORTUNITIES:")
    print("-" * 50)
    for opp in tax_implications["optimization_opportunities"]:
        print(f"• {opp['type']}: {opp['description']}")
        print(f"  Action: {opp['action']}")
        print(f"  Potential Savings: ${opp['potential_savings']:.2f}")
        print()
    
    # Show compliance risks
    print("⚠️  COMPLIANCE RISKS:")
    print("-" * 50)
    for risk in tax_implications["compliance_risks"]:
        print(f"• {risk['type']} ({risk['severity']}): {risk['description']}")
        print(f"  Action: {risk['action']}")
        print()
    
    # Show citations
    print("📚 AUDIT CITATIONS:")
    print("-" * 50)
    for citation in tax_implications["citations"][:3]:  # Show first 3
        print(f"• Transaction {citation['transaction_id']}: {citation['category']}")
        print(f"  IRS Reference: {citation['irs_reference']}")
        print(f"  Deduction Rate: {citation['deduction_rate']:.0%}")
        print()
    
    print(f"📈 Agent Memory:")
    print(f"  - Categorized Transactions: {categorizer.get_memory('categorized_transactions')}")
    print(f"  - Total Deductible: ${categorizer.get_memory('total_deductible'):.2f}")
    print(f"  - Optimization Opportunities: {categorizer.get_memory('optimization_opportunities')}")
    print(f"  - Compliance Risks: {categorizer.get_memory('compliance_risks')}")
    print()
    
    print("🎉 Demo completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())