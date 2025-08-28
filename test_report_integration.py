#!/usr/bin/env python3
"""Integration test for Report Generator Agent with mocked dependencies."""

import sys
import asyncio
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from agents.report_generator import ReportGeneratorAgent
from workflows.state_schemas import WorkflowState, AgentStatus, AnalysisState
from models.base import Transaction


def create_comprehensive_test_data():
    """Create comprehensive test data."""
    transactions = [
        Transaction(
            id="tx1",
            date=datetime(2024, 1, 15),
            amount=Decimal("150.00"),
            description="Office supplies from Staples",
            account_id="acc1",
            counterparty="Staples",
            category="Office Expenses",
            tax_category="Office Expenses",
            source_file="january.csv",
            data_quality_score=0.9,
            data_issues=[]
        ),
        Transaction(
            id="tx2",
            date=datetime(2024, 1, 20),
            amount=Decimal("75.50"),
            description="Business lunch with client",
            account_id="acc1",
            counterparty="Restaurant ABC",
            category="Meals and Entertainment",
            tax_category="Meals and Entertainment",
            source_file="january.csv",
            data_quality_score=0.8,
            data_issues=["Missing receipt"]
        ),
        Transaction(
            id="tx3",
            date=datetime(2024, 2, 1),
            amount=Decimal("500.00"),
            description="Microsoft Office 365 license",
            account_id="acc1",
            counterparty="Microsoft",
            category="Equipment and Depreciation",
            tax_category="Equipment and Depreciation",
            source_file="february.csv",
            data_quality_score=1.0,
            data_issues=[]
        ),
        Transaction(
            id="tx4",
            date=datetime(2024, 2, 15),
            amount=Decimal("1200.00"),
            description="Legal consultation fees",
            account_id="acc1",
            counterparty="Law Firm XYZ",
            category="Professional Services",
            tax_category="Professional Services",
            source_file="february.csv",
            data_quality_score=0.95,
            data_issues=[]
        )
    ]
    
    tax_implications = {
        "total_deductible": 1925.50,
        "optimization_opportunities": [
            {
                "type": "meal_deduction_optimization",
                "description": "Meal expenses of $75.50 qualify for 50% business deduction",
                "potential_savings": 18.88,
                "action": "Ensure proper documentation for business purpose of meals",
                "irs_reference": "IRS Publication 334, Chapter 11"
            },
            {
                "type": "section_179_deduction",
                "description": "Equipment purchases of $500.00 may qualify for Section 179 immediate expensing",
                "potential_savings": 125.00,
                "action": "Consider Section 179 election for immediate deduction vs depreciation",
                "irs_reference": "IRS Publication 334, Chapter 9"
            }
        ],
        "compliance_risks": [
            {
                "type": "uncategorized_transactions",
                "severity": "low",
                "description": "Found 0 uncategorized transactions requiring manual review",
                "transactions": [],
                "action": "Review and properly categorize all transactions for maximum deduction",
                "irs_reference": "Conservative compliance approach"
            }
        ],
        "citations": [
            {
                "transaction_id": "tx1",
                "category": "Office Expenses",
                "irs_reference": "IRS Publication 334, Chapter 8",
                "deduction_rate": 1.0,
                "reasoning": "Categorized as Office Expenses based on description analysis and IRS guidelines"
            }
        ]
    }
    
    workflow_state = WorkflowState(
        workflow_id="comprehensive-test-workflow-456",
        workflow_type="financial_analysis",
        status="running",
        started_at=datetime.utcnow(),
        input_files=["january.csv", "february.csv"],
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


async def test_full_execution():
    """Test full agent execution with mocked dependencies."""
    print("Testing Report Generator Agent full execution...")
    
    try:
        # Create test data
        workflow_state, transactions, tax_implications = create_comprehensive_test_data()
        print("✓ Comprehensive test data created")
        
        # Create agent with mocked dependencies
        with patch('src.agents.report_generator.get_llm_router') as mock_llm_router:
            # Mock LLM router
            mock_llm_instance = Mock()
            mock_llm_instance.generate = AsyncMock(return_value=Mock(
                content="This is a compelling executive summary that demonstrates strategic financial insights and actionable recommendations for business growth and optimization."
            ))
            mock_llm_router.return_value = mock_llm_instance
            
            # Create agent
            agent = ReportGeneratorAgent()
            
            # Mock storage manager methods
            agent.storage_manager.upload_text = AsyncMock(return_value="https://example.com/report.md")
            
            # Mock chart generator methods
            agent.chart_generator.create_pie_chart = AsyncMock(return_value={
                "data": [
                    {"name": "Professional Services", "value": 1200.00, "percentage": 62.3},
                    {"name": "Equipment and Depreciation", "value": 500.00, "percentage": 26.0},
                    {"name": "Office Expenses", "value": 150.00, "percentage": 7.8},
                    {"name": "Meals and Entertainment", "value": 75.50, "percentage": 3.9}
                ],
                "config": {"type": "pie", "title": "Expense Distribution by Category"},
                "type": "pie",
                "title": "Expense Distribution by Category"
            })
            
            agent.chart_generator.create_line_chart = AsyncMock(return_value={
                "data": [
                    {"period": "2024-01", "value": 225.50, "formatted_value": "$225.50"},
                    {"period": "2024-02", "value": 1700.00, "formatted_value": "$1,700.00"}
                ],
                "config": {"type": "line", "title": "Monthly Financial Trends"},
                "type": "line",
                "title": "Monthly Financial Trends"
            })
            
            agent.chart_generator.create_bar_chart = AsyncMock(return_value={
                "data": [
                    {"category": "section_179_deduction", "value": 125.00, "formatted_value": "$125.00"},
                    {"category": "meal_deduction_optimization", "value": 18.88, "formatted_value": "$18.88"}
                ],
                "config": {"type": "bar", "title": "Tax Optimization Opportunities"},
                "type": "bar",
                "title": "Tax Optimization Opportunities"
            })
            
            print("✓ Agent created with mocked dependencies")
            
            # Execute the agent
            result_state = await agent.execute(workflow_state)
            
            print("✓ Agent execution completed successfully")
            
            # Verify results
            assert "report" in result_state
            assert result_state["report"]["status"] == AgentStatus.COMPLETED
            assert "report_id" in result_state["report"]
            assert "content" in result_state["report"]
            assert len(result_state["output_reports"]) == 1
            
            print(f"✓ Report generated with ID: {result_state['report']['report_id']}")
            print(f"✓ Report URL: {result_state['output_reports'][0]}")
            
            # Verify report content structure
            report_content = result_state["report"]["content"]
            assert "# Executive Financial Intelligence Report" in report_content
            assert "## Executive Summary" in report_content
            assert "## Detailed Financial Analysis" in report_content
            assert "## Strategic Recommendations" in report_content
            assert "Professor Elena Castellanos" in report_content
            assert "$1,925.50" in report_content  # Total expenses
            
            print("✓ Report content structure verified")
            
            # Verify visualizations were generated
            visualizations = result_state["report"]["visualizations"]
            assert len(visualizations) == 3
            assert any(viz["type"] == "pie" for viz in visualizations)
            assert any(viz["type"] == "line" for viz in visualizations)
            assert any(viz["type"] == "bar" for viz in visualizations)
            
            print("✓ Visualizations generated successfully")
            
            # Verify LLM was called for narrative generation
            assert mock_llm_instance.generate.call_count == 3  # Three narrative sections
            
            print("✓ LLM narrative generation verified")
            
            # Verify storage was called
            assert agent.storage_manager.upload_text.call_count >= 3  # report, metadata, visualizations
            
            print("✓ Storage operations verified")
            
            # Test report metadata
            metadata = result_state["report"]["metadata"]
            assert metadata["generated_by"] == "Professor Elena Castellanos"
            assert metadata["transaction_count"] == 4
            assert metadata["financial_summary"]["total_expenses"] == 1925.50
            
            print("✓ Report metadata verified")
            
            print("\n🎉 Full execution test passed successfully!")
            
            # Print sample of generated report
            print("\n📄 Sample Report Content:")
            print("=" * 50)
            print(report_content[:500] + "..." if len(report_content) > 500 else report_content)
            print("=" * 50)
            
            return True
            
    except Exception as e:
        print(f"❌ Full execution test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_error_handling():
    """Test error handling scenarios."""
    print("\nTesting error handling scenarios...")
    
    try:
        # Test with empty transactions
        empty_state = WorkflowState(
            workflow_id="empty-test-workflow",
            workflow_type="financial_analysis",
            status="running",
            started_at=datetime.utcnow(),
            input_files=[],
            output_reports=[],
            analysis=AnalysisState(
                transactions=[],
                categories={},
                tax_implications={},
                status=AgentStatus.COMPLETED
            ),
            errors=[],
            warnings=[],
            messages=[],
            config={},
            checkpoint_metadata={}
        )
        
        with patch('src.agents.report_generator.get_llm_router'):
            agent = ReportGeneratorAgent()
            
            try:
                await agent.execute(empty_state)
                print("❌ Expected ValueError for empty transactions")
                return False
            except ValueError as e:
                if "No analyzed transactions available" in str(e):
                    print("✓ Empty transactions error handled correctly")
                else:
                    print(f"❌ Unexpected error message: {e}")
                    return False
        
        print("✓ Error handling tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False


async def main():
    """Run all tests."""
    print("🚀 Starting Report Generator Agent Integration Tests\n")
    
    # Test full execution
    success1 = await test_full_execution()
    
    # Test error handling
    success2 = await test_error_handling()
    
    if success1 and success2:
        print("\n🎉 All integration tests passed successfully!")
        return True
    else:
        print("\n❌ Some tests failed")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)