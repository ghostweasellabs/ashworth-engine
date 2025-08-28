"""Unit tests for the Categorizer Agent."""

import pytest
from datetime import datetime
from decimal import Decimal
from unittest.mock import Mock, patch

from src.agents.categorizer import CategorizerAgent
from src.models.base import Transaction
from src.workflows.state_schemas import WorkflowState, AgentStatus, AnalysisState


class TestCategorizerAgent:
    """Test suite for the Categorizer Agent."""
    
    @pytest.fixture
    def categorizer_agent(self):
        """Create a categorizer agent instance for testing."""
        return CategorizerAgent()
    
    @pytest.fixture
    def sample_transactions(self):
        """Create sample transactions for testing."""
        return [
            Transaction(
                id="tx1",
                date=datetime(2024, 1, 15),
                amount=Decimal("150.00"),
                description="Office supplies from Staples",
                counterparty="Staples",
                source_file="test.csv",
                data_quality_score=0.9,
                data_issues=[]
            ),
            Transaction(
                id="tx2", 
                date=datetime(2024, 1, 20),
                amount=Decimal("75.50"),
                description="Business lunch with client",
                counterparty="Restaurant ABC",
                source_file="test.csv",
                data_quality_score=0.8,
                data_issues=[]
            ),
            Transaction(
                id="tx3",
                date=datetime(2024, 1, 25),
                amount=Decimal("2500.00"),
                description="New laptop for development",
                counterparty="Best Buy",
                source_file="test.csv",
                data_quality_score=0.95,
                data_issues=[]
            ),
            Transaction(
                id="tx4",
                date=datetime(2024, 1, 30),
                amount=Decimal("12000.00"),
                description="Large equipment purchase",
                counterparty="Equipment Corp",
                source_file="test.csv",
                data_quality_score=0.9,
                data_issues=[]
            )
        ]
    
    @pytest.fixture
    def workflow_state_with_transactions(self, sample_transactions):
        """Create workflow state with sample transactions."""
        return WorkflowState(
            workflow_id="test-workflow",
            analysis=AnalysisState(
                transactions=[tx.model_dump() for tx in sample_transactions],
                status=AgentStatus.COMPLETED
            )
        )
    
    def test_agent_initialization(self, categorizer_agent):
        """Test that the categorizer agent initializes correctly."""
        assert categorizer_agent.get_agent_id() == "categorizer"
        assert categorizer_agent.personality.name == "Clarke Pemberton, JD, CPA"
        assert len(categorizer_agent.irs_categories) > 0
        assert "office_expenses" in categorizer_agent.irs_categories
        assert "meals" in categorizer_agent.irs_categories
        assert "equipment" in categorizer_agent.irs_categories
    
    def test_irs_categories_structure(self, categorizer_agent):
        """Test that IRS categories have proper structure."""
        for category_key, category_info in categorizer_agent.irs_categories.items():
            assert "name" in category_info
            assert "deduction_rate" in category_info
            assert "keywords" in category_info
            assert "irs_reference" in category_info
            assert isinstance(category_info["deduction_rate"], Decimal)
            assert isinstance(category_info["keywords"], list)
    
    def test_determine_category_office_supplies(self, categorizer_agent):
        """Test categorization of office supplies."""
        transaction = Transaction(
            id="test",
            date=datetime.now(),
            amount=Decimal("50.00"),
            description="Office supplies and paper",
            source_file="test.csv",
            data_quality_score=0.9,
            data_issues=[]
        )
        
        category_info = categorizer_agent._determine_category(transaction)
        assert category_info["name"] == "Office Expenses"
        assert category_info["deduction_rate"] == Decimal("1.0")
    
    def test_determine_category_meals(self, categorizer_agent):
        """Test categorization of meal expenses."""
        transaction = Transaction(
            id="test",
            date=datetime.now(),
            amount=Decimal("75.00"),
            description="Business lunch meeting",
            source_file="test.csv",
            data_quality_score=0.9,
            data_issues=[]
        )
        
        category_info = categorizer_agent._determine_category(transaction)
        assert category_info["name"] == "Meals and Entertainment"
        assert category_info["deduction_rate"] == Decimal("0.5")  # 50% deduction
    
    def test_determine_category_equipment(self, categorizer_agent):
        """Test categorization of equipment purchases."""
        transaction = Transaction(
            id="test",
            date=datetime.now(),
            amount=Decimal("1500.00"),
            description="New laptop computer equipment",
            source_file="test.csv",
            data_quality_score=0.9,
            data_issues=[]
        )
        
        category_info = categorizer_agent._determine_category(transaction)
        assert category_info["name"] == "Equipment and Depreciation"
        assert category_info["deduction_rate"] == Decimal("1.0")
    
    def test_determine_category_uncategorized(self, categorizer_agent):
        """Test that unclear transactions are marked as uncategorized."""
        transaction = Transaction(
            id="test",
            date=datetime.now(),
            amount=Decimal("100.00"),
            description="Miscellaneous expense xyz",
            source_file="test.csv",
            data_quality_score=0.9,
            data_issues=[]
        )
        
        category_info = categorizer_agent._determine_category(transaction)
        assert category_info["name"] == "Uncategorized - Requires Review"
        assert category_info["deduction_rate"] == Decimal("0.0")  # Conservative approach
    
    @pytest.mark.asyncio
    async def test_categorize_transactions(self, categorizer_agent, sample_transactions):
        """Test the transaction categorization process."""
        categorized = await categorizer_agent._categorize_transactions(sample_transactions)
        
        assert len(categorized) == len(sample_transactions)
        
        # Check that all transactions have categories assigned
        for tx in categorized:
            assert tx.category is not None
            assert tx.tax_category is not None
        
        # Check specific categorizations
        office_tx = next(tx for tx in categorized if "office" in tx.description.lower())
        assert office_tx.tax_category == "Office Expenses"
        
        meal_tx = next(tx for tx in categorized if "lunch" in tx.description.lower())
        assert meal_tx.tax_category == "Meals and Entertainment"
        
        equipment_tx = next(tx for tx in categorized if "laptop" in tx.description.lower())
        assert equipment_tx.tax_category == "Equipment and Depreciation"
    
    def test_analyze_tax_optimization(self, categorizer_agent, sample_transactions):
        """Test tax optimization analysis."""
        # Create transactions with meal expenses
        meal_transactions = [
            Transaction(
                id=f"meal_{i}",
                date=datetime.now(),
                amount=Decimal("100.00"),
                description="Business meal",
                tax_category="Meals and Entertainment",
                source_file="test.csv",
                data_quality_score=0.9,
                data_issues=[]
            ) for i in range(15)  # $1500 total in meals
        ]
        
        opportunities = categorizer_agent._analyze_tax_optimization(meal_transactions)
        
        # Should identify meal deduction optimization
        meal_opportunity = next((opp for opp in opportunities if opp["type"] == "meal_deduction_optimization"), None)
        assert meal_opportunity is not None
        assert meal_opportunity["potential_savings"] > 0
        assert "50%" in meal_opportunity["description"]
    
    def test_assess_compliance_risks_large_transactions(self, categorizer_agent):
        """Test compliance risk assessment for large transactions."""
        large_transactions = [
            Transaction(
                id="large1",
                date=datetime.now(),
                amount=Decimal("15000.00"),  # Above $10k threshold
                description="Large equipment purchase",
                source_file="test.csv",
                data_quality_score=0.9,
                data_issues=[]
            )
        ]
        
        risks = categorizer_agent._assess_compliance_risks(large_transactions)
        
        # Should flag Form 8300 requirement
        form_8300_risk = next((risk for risk in risks if risk["type"] == "form_8300_requirement"), None)
        assert form_8300_risk is not None
        assert form_8300_risk["severity"] == "high"
        assert "Form 8300" in form_8300_risk["description"]
    
    def test_assess_compliance_risks_round_numbers(self, categorizer_agent):
        """Test compliance risk assessment for round number patterns."""
        # Create mostly round-number transactions
        round_transactions = [
            Transaction(
                id=f"round_{i}",
                date=datetime.now(),
                amount=Decimal(f"{(i+1)*100}.00"),  # $100, $200, $300, etc.
                description=f"Transaction {i+1}",
                source_file="test.csv",
                data_quality_score=0.9,
                data_issues=[]
            ) for i in range(10)
        ]
        
        risks = categorizer_agent._assess_compliance_risks(round_transactions)
        
        # Should flag round number pattern
        round_risk = next((risk for risk in risks if risk["type"] == "round_number_pattern"), None)
        assert round_risk is not None
        assert round_risk["severity"] == "medium"
    
    def test_assess_compliance_risks_uncategorized(self, categorizer_agent):
        """Test compliance risk assessment for uncategorized transactions."""
        uncategorized_transactions = [
            Transaction(
                id="uncat1",
                date=datetime.now(),
                amount=Decimal("100.00"),
                description="Unknown expense",
                tax_category="Uncategorized - Requires Review",
                source_file="test.csv",
                data_quality_score=0.9,
                data_issues=[]
            )
        ]
        
        risks = categorizer_agent._assess_compliance_risks(uncategorized_transactions)
        
        # Should flag uncategorized transactions
        uncat_risk = next((risk for risk in risks if risk["type"] == "uncategorized_transactions"), None)
        assert uncat_risk is not None
        assert uncat_risk["severity"] == "low"
    
    def test_calculate_tax_metrics(self, categorizer_agent):
        """Test tax metrics calculation."""
        transactions = [
            Transaction(
                id="tx1",
                date=datetime.now(),
                amount=Decimal("1000.00"),
                description="Office supplies",
                tax_category="Office Expenses",
                source_file="test.csv",
                data_quality_score=0.9,
                data_issues=[]
            ),
            Transaction(
                id="tx2",
                date=datetime.now(),
                amount=Decimal("200.00"),
                description="Business meal",
                tax_category="Meals and Entertainment",
                source_file="test.csv",
                data_quality_score=0.9,
                data_issues=[]
            )
        ]
        
        metrics = categorizer_agent._calculate_tax_metrics(transactions)
        
        assert metrics["total_expenses"] == Decimal("1200.00")
        # Office: $1000 * 1.0 + Meals: $200 * 0.5 = $1100 deductible
        assert metrics["total_deductible"] == Decimal("1100.00")
        # Estimated savings: $1100 * 0.25 = $275
        assert metrics["estimated_savings"] == Decimal("275.00")
    
    def test_generate_citation_report(self, categorizer_agent):
        """Test citation report generation."""
        # Add some test citations
        categorizer_agent.citations = [
            {
                "transaction_id": "tx1",
                "category": "Office Expenses",
                "irs_reference": "IRS Publication 334, Chapter 8",
                "deduction_rate": 1.0,
                "reasoning": "Test reasoning"
            }
        ]
        
        citations = categorizer_agent._generate_citation_report()
        
        assert len(citations) == 1
        assert citations[0]["transaction_id"] == "tx1"
        assert citations[0]["irs_reference"] == "IRS Publication 334, Chapter 8"
    
    def test_get_category_key(self, categorizer_agent):
        """Test category key lookup."""
        assert categorizer_agent._get_category_key("Office Expenses") == "office_expenses"
        assert categorizer_agent._get_category_key("Meals and Entertainment") == "meals"
        assert categorizer_agent._get_category_key("Unknown Category") == "uncategorized"
    
    def test_generate_category_summary(self, categorizer_agent):
        """Test category summary generation."""
        transactions = [
            Transaction(
                id="tx1",
                date=datetime.now(),
                amount=Decimal("100.00"),
                description="Test",
                tax_category="Office Expenses",
                source_file="test.csv",
                data_quality_score=0.9,
                data_issues=[]
            ),
            Transaction(
                id="tx2",
                date=datetime.now(),
                amount=Decimal("50.00"),
                description="Test",
                tax_category="Meals and Entertainment",
                source_file="test.csv",
                data_quality_score=0.9,
                data_issues=[]
            )
        ]
        
        summary = categorizer_agent._generate_category_summary(transactions)
        
        assert summary["tx1"] == "Office Expenses"
        assert summary["tx2"] == "Meals and Entertainment"
    
    @pytest.mark.asyncio
    async def test_execute_success(self, categorizer_agent, workflow_state_with_transactions):
        """Test successful execution of the categorizer agent."""
        result_state = await categorizer_agent.execute(workflow_state_with_transactions)
        
        # Check that analysis state was updated
        analysis = result_state["analysis"]
        assert analysis["status"] == AgentStatus.COMPLETED
        assert len(analysis["transactions"]) == 4
        assert "tax_implications" in analysis
        
        # Check tax implications structure
        tax_implications = analysis["tax_implications"]
        assert "total_deductible" in tax_implications
        assert "optimization_opportunities" in tax_implications
        assert "compliance_risks" in tax_implications
        assert "citations" in tax_implications
        assert "tax_savings_estimate" in tax_implications
    
    @pytest.mark.asyncio
    async def test_execute_no_transactions(self, categorizer_agent):
        """Test execution with no transactions."""
        empty_state = WorkflowState(
            workflow_id="test-workflow",
            analysis=AnalysisState(
                transactions=[],
                status=AgentStatus.COMPLETED
            )
        )
        
        with pytest.raises(ValueError, match="No processed transactions available"):
            await categorizer_agent.execute(empty_state)
    
    @pytest.mark.asyncio
    async def test_execute_missing_analysis_state(self, categorizer_agent):
        """Test execution with missing analysis state."""
        empty_state = WorkflowState(workflow_id="test-workflow")
        
        with pytest.raises(ValueError, match="No processed transactions available"):
            await categorizer_agent.execute(empty_state)
    
    def test_optimization_rules_configuration(self, categorizer_agent):
        """Test that optimization rules are properly configured."""
        rules = categorizer_agent.optimization_rules
        
        assert "large_transaction_threshold" in rules
        assert rules["large_transaction_threshold"] == Decimal("10000.00")
        assert "meal_deduction_limit" in rules
        assert rules["meal_deduction_limit"] == Decimal("0.5")
    
    def test_audit_triggers_configuration(self, categorizer_agent):
        """Test that audit triggers are properly configured."""
        triggers = categorizer_agent.audit_triggers
        
        assert "cash_transaction_limit" in triggers
        assert triggers["cash_transaction_limit"] == Decimal("10000.00")
        assert "round_number_percentage" in triggers
        assert triggers["round_number_percentage"] == 0.8
    
    def test_meal_deduction_rate(self, categorizer_agent):
        """Test that meal deduction rate is correctly set to 50%."""
        meals_category = categorizer_agent.irs_categories["meals"]
        assert meals_category["deduction_rate"] == Decimal("0.5")
        assert "meal" in meals_category["name"].lower()
    
    def test_conservative_compliance_approach(self, categorizer_agent):
        """Test that uncategorized transactions have 0% deduction rate (conservative)."""
        uncategorized_category = categorizer_agent.irs_categories["uncategorized"]
        assert uncategorized_category["deduction_rate"] == Decimal("0.0")
        assert "review" in uncategorized_category["name"].lower()