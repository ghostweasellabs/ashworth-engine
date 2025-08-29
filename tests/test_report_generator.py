"""Tests for the Report Generator Agent."""

import pytest
import asyncio
from datetime import datetime
from decimal import Decimal
from unittest.mock import Mock, AsyncMock, patch

from src.agents.report_generator import ReportGeneratorAgent
from src.workflows.state_schemas import WorkflowState, AgentStatus, AnalysisState
from src.models.base import Transaction


@pytest.fixture
def sample_transactions():
    """Create sample transactions for testing."""
    return [
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
        ),
        Transaction(
            id="tx3",
            date=datetime(2024, 2, 1),
            amount=Decimal("500.00"),
            description="Software license",
            account_id="acc1",
            counterparty="Microsoft",
            category="Equipment and Depreciation",
            tax_category="Equipment and Depreciation",
            source_file="test.csv",
            data_quality_score=1.0,
            data_issues=[]
        )
    ]


@pytest.fixture
def sample_tax_implications():
    """Create sample tax implications for testing."""
    return {
        "total_deductible": 725.50,
        "optimization_opportunities": [
            {
                "type": "meal_deduction_optimization",
                "description": "Meal expenses qualify for 50% business deduction",
                "potential_savings": 18.88,
                "action": "Ensure proper documentation",
                "irs_reference": "IRS Publication 334, Chapter 11"
            }
        ],
        "compliance_risks": [
            {
                "type": "uncategorized_transactions",
                "severity": "low",
                "description": "Some transactions need manual review",
                "action": "Review and categorize",
                "irs_reference": "Conservative compliance approach"
            }
        ],
        "citations": []
    }


@pytest.fixture
def sample_workflow_state(sample_transactions, sample_tax_implications):
    """Create a sample workflow state for testing."""
    return WorkflowState(
        workflow_id="test-workflow-123",
        workflow_type="financial_analysis",
        status="running",
        started_at=datetime.utcnow(),
        input_files=["test.csv"],
        output_reports=[],
        analysis=AnalysisState(
            transactions=[tx.model_dump() for tx in sample_transactions],
            categories={tx.id: tx.tax_category for tx in sample_transactions},
            tax_implications=sample_tax_implications,
            status=AgentStatus.COMPLETED
        ),
        errors=[],
        warnings=[],
        messages=[],
        config={},
        checkpoint_metadata={}
    )


class TestReportGeneratorAgent:
    """Test cases for the Report Generator Agent."""
    
    @pytest.fixture
    def agent(self):
        """Create a Report Generator Agent instance for testing."""
        with patch('src.agents.report_generator.get_llm_router') as mock_llm, \
             patch('src.agents.report_generator.SupabaseStorageManager') as mock_storage, \
             patch('src.agents.report_generator.ChartGenerator') as mock_charts:
            
            # Mock LLM router
            mock_llm_instance = Mock()
            mock_llm_instance.generate = AsyncMock(return_value=Mock(content="Generated narrative content"))
            mock_llm.return_value = mock_llm_instance
            
            # Mock storage manager
            mock_storage_instance = Mock()
            mock_storage_instance.upload_text = AsyncMock(return_value="https://example.com/report.md")
            mock_storage.return_value = mock_storage_instance
            
            # Mock chart generator
            mock_charts_instance = Mock()
            mock_charts_instance.create_pie_chart = AsyncMock(return_value={
                "data": [{"name": "Office Expenses", "value": 150.00}],
                "config": {"type": "pie"},
                "type": "pie",
                "title": "Test Chart"
            })
            mock_charts_instance.create_line_chart = AsyncMock(return_value={
                "data": [{"period": "2024-01", "value": 225.50}],
                "config": {"type": "line"},
                "type": "line",
                "title": "Monthly Trends"
            })
            mock_charts_instance.create_bar_chart = AsyncMock(return_value={
                "data": [{"category": "meal_deduction", "value": 18.88}],
                "config": {"type": "bar"},
                "type": "bar",
                "title": "Tax Optimization"
            })
            mock_charts.return_value = mock_charts_instance
            
            agent = ReportGeneratorAgent()
            agent.storage_manager = mock_storage_instance
            agent.chart_generator = mock_charts_instance
            agent.llm_router = mock_llm_instance
            
            return agent
    
    def test_agent_initialization(self, agent):
        """Test that the agent initializes correctly."""
        assert agent.get_agent_id() == "report_generator"
        assert agent.personality.name == "Professor Elena Castellanos"
        assert "executive communication" in agent.personality.expertise_areas
    
    def test_calculate_comprehensive_metrics(self, agent, sample_transactions, sample_tax_implications):
        """Test financial metrics calculation."""
        metrics = agent._calculate_comprehensive_metrics(sample_transactions, sample_tax_implications)
        
        assert metrics.total_expenses == Decimal("725.50")
        assert metrics.total_revenue == Decimal("0")  # No negative amounts in sample
        assert metrics.net_income == Decimal("-725.50")
        assert metrics.tax_deductible_amount == Decimal("725.50")
        assert len(metrics.expense_categories) == 3
        assert "Office Expenses" in metrics.expense_categories
    
    @pytest.mark.asyncio
    async def test_generate_visualizations(self, agent, sample_transactions, sample_tax_implications):
        """Test visualization generation."""
        financial_metrics = agent._calculate_comprehensive_metrics(sample_transactions, sample_tax_implications)
        
        visualizations = await agent._generate_visualizations(
            sample_transactions, financial_metrics, sample_tax_implications
        )
        
        assert len(visualizations) == 3  # pie, line, bar charts
        assert any(viz["type"] == "pie" for viz in visualizations)
        assert any(viz["type"] == "line" for viz in visualizations)
        assert any(viz["type"] == "bar" for viz in visualizations)
        
        # Verify chart generator was called
        agent.chart_generator.create_pie_chart.assert_called_once()
        agent.chart_generator.create_line_chart.assert_called_once()
        agent.chart_generator.create_bar_chart.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_generate_executive_narrative(self, agent, sample_transactions, sample_tax_implications):
        """Test executive narrative generation."""
        financial_metrics = agent._calculate_comprehensive_metrics(sample_transactions, sample_tax_implications)
        visualizations = []
        
        narrative = await agent._generate_executive_narrative(
            sample_transactions, financial_metrics, sample_tax_implications, visualizations
        )
        
        assert "executive_summary" in narrative
        assert "detailed_analysis" in narrative
        assert "strategic_recommendations" in narrative
        assert "quality_score" in narrative
        assert narrative["quality_score"] >= 0.0
        
        # Verify LLM was called for narrative generation
        assert agent.llm_router.generate.call_count == 3  # Three sections
    
    def test_structure_complete_report(self, agent, sample_transactions, sample_tax_implications):
        """Test report structuring."""
        financial_metrics = agent._calculate_comprehensive_metrics(sample_transactions, sample_tax_implications)
        
        narrative_content = {
            "executive_summary": "Test executive summary",
            "detailed_analysis": "Test detailed analysis",
            "strategic_recommendations": "Test recommendations",
            "quality_score": 4.2,
            "generation_metadata": {"total_words": 100}
        }
        
        visualizations = [
            {
                "id": "test_chart",
                "type": "pie",
                "title": "Test Chart",
                "insights": "Test insights"
            }
        ]
        
        report = agent._structure_complete_report(
            narrative_content, visualizations, financial_metrics, sample_tax_implications
        )
        
        assert "# Executive Financial Intelligence Report" in report
        assert "## Executive Summary" in report
        assert "## Detailed Financial Analysis" in report
        assert "## Strategic Recommendations" in report
        assert "Test executive summary" in report
        assert "Professor Elena Castellanos" in report
        assert "$725.50" in report  # Total expenses
    
    def test_generate_report_metadata(self, agent, sample_transactions):
        """Test report metadata generation."""
        financial_metrics = agent._calculate_comprehensive_metrics(sample_transactions, {})
        
        metadata = agent._generate_report_metadata(
            "test-workflow-123", len(sample_transactions), financial_metrics
        )
        
        assert metadata["workflow_id"] == "test-workflow-123"
        assert metadata["generated_by"] == "Professor Elena Castellanos"
        assert metadata["agent_id"] == "report_generator"
        assert metadata["transaction_count"] == 3
        assert "financial_summary" in metadata
        assert metadata["financial_summary"]["total_expenses"] == 725.50
    
    @pytest.mark.asyncio
    async def test_store_report_with_versioning(self, agent):
        """Test report storage with versioning."""
        report_content = "# Test Report\nThis is a test report."
        visualizations = [{"id": "test", "type": "pie"}]
        metadata = {
            "report_id": "test-report-123",
            "workflow_id": "test-workflow-123",
            "generated_at": datetime.utcnow().isoformat()
        }
        
        result = await agent._store_report_with_versioning(
            report_content, visualizations, metadata
        )
        
        assert "report_id" in result
        assert "storage_path" in result
        assert "public_url" in result
        assert result["report_id"] == "test-report-123"
        
        # Verify storage manager was called
        assert agent.storage_manager.upload_text.call_count >= 3  # report, metadata, visualizations
    
    def test_aggregate_monthly_data(self, agent, sample_transactions):
        """Test monthly data aggregation."""
        monthly_data = agent._aggregate_monthly_data(sample_transactions)
        
        assert "2024-01" in monthly_data
        assert "2024-02" in monthly_data
        assert monthly_data["2024-01"] == 225.50  # Two January transactions
        assert monthly_data["2024-02"] == 500.00  # One February transaction
    
    def test_generate_chart_insights(self, agent):
        """Test chart insights generation."""
        # Test expense categories insights
        expense_data = {"Office Expenses": 150.00, "Meals": 75.50}
        insights = agent._generate_chart_insights("expense_categories", expense_data)
        
        assert "Office Expenses" in insights
        assert "66.5%" in insights  # 150/(150+75.5) = 66.5%
        
        # Test monthly trends insights
        monthly_data = {"2024-01": 100.00, "2024-02": 120.00}
        insights = agent._generate_chart_insights("monthly_trends", monthly_data)
        
        assert "increased" in insights
        assert "20.0%" in insights  # (120-100)/100 = 20%
    
    def test_assess_narrative_quality(self, agent):
        """Test narrative quality assessment."""
        good_section = "This strategic analysis reveals optimization opportunities for executive implementation."
        poor_section = "Short."
        
        quality_score = agent._assess_narrative_quality(good_section, poor_section)
        
        assert 0.0 <= quality_score <= 5.0
        assert quality_score > 0  # Should have some positive score
    
    @pytest.mark.asyncio
    async def test_execute_success(self, agent, sample_workflow_state):
        """Test successful agent execution."""
        # Mock the storage result
        agent._store_report_with_versioning = AsyncMock(return_value={
            "report_id": "test-report-123",
            "storage_path": "reports/test-workflow-123/test-report-123",
            "public_url": "https://example.com/report.md",
            "metadata_url": "https://example.com/metadata.json",
            "visualizations_url": "https://example.com/viz.json"
        })
        
        result_state = await agent.execute(sample_workflow_state)
        
        assert "report" in result_state
        assert result_state["report"]["status"] == AgentStatus.COMPLETED
        assert result_state["report"]["report_id"] == "test-report-123"
        assert len(result_state["output_reports"]) == 1
        assert result_state["output_reports"][0] == "https://example.com/report.md"
    
    @pytest.mark.asyncio
    async def test_execute_no_transactions_error(self, agent):
        """Test agent execution with no transactions."""
        empty_state = WorkflowState(
            workflow_id="test-workflow-123",
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
        
        with pytest.raises(ValueError, match="No analyzed transactions available"):
            await agent.execute(empty_state)
    
    def test_fallback_narrative_generation(self, agent):
        """Test fallback narrative generation when LLM fails."""
        context = {
            "transaction_count": 3,
            "total_expenses": 725.50,
            "total_revenue": 0.0,
            "net_income": -725.50,
            "tax_deductible": 725.50,
            "top_categories": {"Office Expenses": 150.00},
            "optimization_opportunities": 1,
            "compliance_risks": 1
        }
        
        # Test executive summary fallback
        summary = agent._generate_fallback_narrative("executive_summary", context)
        assert "Executive Summary" in summary
        assert "725.50" in summary
        assert "3 transactions" in summary
        
        # Test detailed analysis fallback
        analysis = agent._generate_fallback_narrative("detailed_analysis", context)
        assert "Detailed Financial Analysis" in analysis
        assert "Office Expenses" in analysis
        
        # Test strategic recommendations fallback
        recommendations = agent._generate_fallback_narrative("strategic_recommendations", context)
        assert "Strategic Recommendations" in recommendations
        assert "Immediate Actions" in recommendations