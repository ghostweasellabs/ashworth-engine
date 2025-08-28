#!/usr/bin/env python3
"""Simple test script to verify Report Generator Agent implementation."""

import sys
import asyncio
from datetime import datetime
from decimal import Decimal
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from agents.report_generator import ReportGeneratorAgent
from workflows.state_schemas import WorkflowState, AgentStatus, AnalysisState
from models.base import Transaction


def create_sample_data():
    """Create sample data for testing."""
    transactions = [
        Transaction(
            id="tx1",
            date=datetime(2024, 1, 15),
            amount=Decimal("150.00"),
            description="Office supplies",
            account_id="acc1",
            counterparty="Staples",
            category="Office Expenses",
            tax_category="Office Expenses",
            source_file="test.csv",
            data_quality_score=0.9,
            data_issues=[]
        ),
        Transaction(
            id="tx2",
            date=datetime(2024, 1, 20),
            amount=Decimal("75.50"),
            description="Business lunch",
            account_id="acc1",
            counterparty="Restaurant ABC",
            category="Meals and Entertainment",
            tax_category="Meals and Entertainment",
            source_file="test.csv",
            data_quality_score=0.8,
            data_issues=["Missing receipt"]
        )
    ]
    
    tax_implications = {
        "total_deductible": 225.50,
        "optimization_opportunities": [
            {
                "type": "meal_deduction_optimization",
                "description": "Meal expenses qualify for 50% business deduction",
                "potential_savings": 18.88,
                "action": "Ensure proper documentation",
                "irs_reference": "IRS Publication 334, Chapter 11"
            }
        ],
        "compliance_risks": [],
        "citations": []
    }
    
    workflow_state = WorkflowState(
        workflow_id="test-workflow-123",
        workflow_type="financial_analysis",
        status="running",
        started_at=datetime.utcnow(),
        input_files=["test.csv"],
        output_reports=[],
        analysis=AnalysisState(
            transactions=[tx.model_dump() for tx in transactions],
            categories={tx.id: tx.tax_category for tx in transactions},
            tax_implications=tax_implications,
            status=AgentStatus.COMPLETED
        ),
        errors=[],
        warnings=[],
        messages=[],
        config={},
        checkpoint_metadata={}
    )
    
    return workflow_state, transactions, tax_implications


def test_basic_functionality():
    """Test basic functionality without external dependencies."""
    print("Testing Report Generator Agent basic functionality...")
    
    try:
        # Create agent (this will fail due to missing dependencies, but we can test the class structure)
        from agents.report_generator import ReportGeneratorAgent
        
        print("✓ ReportGeneratorAgent class imported successfully")
        
        # Test data creation
        workflow_state, transactions, tax_implications = create_sample_data()
        print("✓ Sample data created successfully")
        
        # Test basic methods that don't require external dependencies
        agent = ReportGeneratorAgent.__new__(ReportGeneratorAgent)  # Create without __init__
        
        # Test financial metrics calculation
        financial_metrics = agent._calculate_comprehensive_metrics(transactions, tax_implications)
        print(f"✓ Financial metrics calculated: ${financial_metrics.total_expenses}")
        
        # Test monthly data aggregation
        monthly_data = agent._aggregate_monthly_data(transactions)
        print(f"✓ Monthly data aggregated: {monthly_data}")
        
        # Test chart insights generation
        expense_data = {"Office Expenses": 150.00, "Meals": 75.50}
        insights = agent._generate_chart_insights("expense_categories", expense_data)
        print(f"✓ Chart insights generated: {insights[:100]}...")
        
        # Test narrative quality assessment
        test_narrative = "This strategic analysis reveals optimization opportunities for executive implementation."
        quality_score = agent._assess_narrative_quality(test_narrative)
        print(f"✓ Narrative quality assessed: {quality_score}")
        
        # Test report metadata generation
        metadata = agent._generate_report_metadata("test-workflow", 2, financial_metrics)
        print(f"✓ Report metadata generated: {metadata['report_type']}")
        
        # Test fallback narrative generation
        context = {
            "transaction_count": 2,
            "total_expenses": 225.50,
            "total_revenue": 0.0,
            "net_income": -225.50,
            "tax_deductible": 225.50,
            "top_categories": {"Office Expenses": 150.00},
            "optimization_opportunities": 1,
            "compliance_risks": 0
        }
        
        fallback_summary = agent._generate_fallback_narrative("executive_summary", context)
        print(f"✓ Fallback narrative generated: {len(fallback_summary)} characters")
        
        print("\n🎉 All basic functionality tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_basic_functionality()
    sys.exit(0 if success else 1)