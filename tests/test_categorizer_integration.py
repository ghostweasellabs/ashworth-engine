"""Integration tests for the Categorizer Agent with real-world data scenarios."""

import pytest
from datetime import datetime
from decimal import Decimal
from unittest.mock import Mock, patch

from src.agents.categorizer import CategorizerAgent
from src.models.base import Transaction
from src.workflows.state_schemas import WorkflowState, AgentStatus, AnalysisState


class TestCategorizerIntegration:
    """Integration test suite for the Categorizer Agent."""
    
    @pytest.fixture
    def categorizer_agent(self):
        """Create a categorizer agent instance for testing."""
        return CategorizerAgent()
    
    @pytest.fixture
    def real_world_transactions(self):
        """Create realistic transaction data for integration testing."""
        return [
            # Office supplies
            Transaction(
                id="tx1",
                date=datetime(2024, 1, 15),
                amount=Decimal("89.47"),
                description="STAPLES OFFICE SUPPLIES",
                counterparty="Staples Inc",
                source_file="bank_export.csv",
                data_quality_score=0.95,
                data_issues=[]
            ),
            # Business meal
            Transaction(
                id="tx2",
                date=datetime(2024, 1, 18),
                amount=Decimal("127.83"),
                description="RESTAURANT MEETING CLIENT JONES",
                counterparty="The Capital Grille",
                source_file="bank_export.csv",
                data_quality_score=0.9,
                data_issues=["Missing receipt number"]
            ),
            # Travel expense
            Transaction(
                id="tx3",
                date=datetime(2024, 1, 22),
                amount=Decimal("456.78"),
                description="UNITED AIRLINES BUSINESS TRIP NYC",
                counterparty="United Airlines",
                source_file="credit_card.csv",
                data_quality_score=0.98,
                data_issues=[]
            ),
            # Equipment purchase
            Transaction(
                id="tx4",
                date=datetime(2024, 1, 25),
                amount=Decimal("2847.99"),
                description="DELL LAPTOP DEVELOPMENT TEAM",
                counterparty="Dell Technologies",
                source_file="bank_export.csv",
                data_quality_score=0.92,
                data_issues=[]
            ),
            # Professional services
            Transaction(
                id="tx5",
                date=datetime(2024, 1, 28),
                amount=Decimal("3500.00"),
                description="LEGAL SERVICES CONTRACT REVIEW",
                counterparty="Smith & Associates Law",
                source_file="bank_export.csv",
                data_quality_score=0.88,
                data_issues=["Round amount - verify authenticity"]
            ),
            # Utilities
            Transaction(
                id="tx6",
                date=datetime(2024, 2, 1),
                amount=Decimal("234.56"),
                description="ELECTRIC BILL OFFICE SPACE",
                counterparty="ConEd",
                source_file="bank_export.csv",
                data_quality_score=0.94,
                data_issues=[]
            ),
            # Large transaction requiring Form 8300
            Transaction(
                id="tx7",
                date=datetime(2024, 2, 5),
                amount=Decimal("15000.00"),
                description="EQUIPMENT PURCHASE MANUFACTURING",
                counterparty="Industrial Equipment Corp",
                source_file="bank_export.csv",
                data_quality_score=0.85,
                data_issues=["Large round amount", "Requires Form 8300"]
            ),
            # Ambiguous transaction
            Transaction(
                id="tx8",
                date=datetime(2024, 2, 8),
                amount=Decimal("299.99"),
                description="MISC EXPENSE VENDOR XYZ",
                counterparty="Unknown Vendor",
                source_file="bank_export.csv",
                data_quality_score=0.6,
                data_issues=["Unclear description", "Unknown vendor"]
            ),
            # Marketing expense
            Transaction(
                id="tx9",
                date=datetime(2024, 2, 10),
                amount=Decimal("1250.00"),
                description="GOOGLE ADS MARKETING CAMPAIGN",
                counterparty="Google LLC",
                source_file="credit_card.csv",
                data_quality_score=0.96,
                data_issues=[]
            ),
            # Insurance
            Transaction(
                id="tx10",
                date=datetime(2024, 2, 12),
                amount=Decimal("567.89"),
                description="LIABILITY INSURANCE PREMIUM",
                counterparty="State Farm Insurance",
                source_file="bank_export.csv",
                data_quality_score=0.93,
                data_issues=[]
            )
        ]
    
    @pytest.fixture
    def workflow_state_with_real_data(self, real_world_transactions):
        """Create workflow state with realistic transaction data."""
        return WorkflowState(
            workflow_id="integration-test-workflow",
            analysis=AnalysisState(
                transactions=[tx.model_dump() for tx in real_world_transactions],
                status=AgentStatus.COMPLETED,
                metrics={},
                anomalies=[],
                compliance_issues=[]
            )
        )
    
    @pytest.mark.asyncio
    async def test_full_categorization_workflow(self, categorizer_agent, workflow_state_with_real_data):
        """Test complete categorization workflow with realistic data."""
        result_state = await categorizer_agent.execute(workflow_state_with_real_data)
        
        # Verify workflow completed successfully
        analysis = result_state["analysis"]
        assert analysis["status"] == AgentStatus.COMPLETED
        assert len(analysis["transactions"]) == 10
        
        # Verify all transactions have categories
        for tx_data in analysis["transactions"]:
            assert tx_data["category"] is not None
            assert tx_data["tax_category"] is not None
        
        # Verify tax implications are calculated
        tax_implications = analysis["tax_implications"]
        assert "total_deductible" in tax_implications
        assert "optimization_opportunities" in tax_implications
        assert "compliance_risks" in tax_implications
        assert "citations" in tax_implications
        assert "tax_savings_estimate" in tax_implications
    
    @pytest.mark.asyncio
    async def test_specific_categorizations(self, categorizer_agent, real_world_transactions):
        """Test that specific transaction types are categorized correctly."""
        categorized = await categorizer_agent._categorize_transactions(real_world_transactions)
        
        # Find specific transactions and verify categorization
        staples_tx = next(tx for tx in categorized if "STAPLES" in tx.description)
        assert staples_tx.tax_category == "Office Expenses"
        
        restaurant_tx = next(tx for tx in categorized if "RESTAURANT" in tx.description)
        assert restaurant_tx.tax_category == "Meals and Entertainment"
        
        airline_tx = next(tx for tx in categorized if "UNITED AIRLINES" in tx.description)
        assert airline_tx.tax_category == "Travel Expenses"
        
        laptop_tx = next(tx for tx in categorized if "LAPTOP" in tx.description)
        assert laptop_tx.tax_category == "Equipment and Depreciation"
        
        legal_tx = next(tx for tx in categorized if "LEGAL" in tx.description)
        assert legal_tx.tax_category == "Professional Services"
        
        electric_tx = next(tx for tx in categorized if "ELECTRIC" in tx.description)
        assert electric_tx.tax_category == "Utilities"
        
        ads_tx = next(tx for tx in categorized if "GOOGLE ADS" in tx.description)
        assert ads_tx.tax_category == "Marketing and Advertising"
        
        insurance_tx = next(tx for tx in categorized if "INSURANCE" in tx.description)
        assert insurance_tx.tax_category == "Insurance"
        
        # Ambiguous transaction should be uncategorized
        misc_tx = next(tx for tx in categorized if "MISC EXPENSE" in tx.description)
        assert misc_tx.tax_category == "Uncategorized - Requires Review"
    
    def test_tax_optimization_analysis_realistic(self, categorizer_agent, real_world_transactions):
        """Test tax optimization analysis with realistic transaction amounts."""
        # Categorize transactions first
        categorized = []
        for tx in real_world_transactions:
            category_info = categorizer_agent._determine_category(tx)
            tx.tax_category = category_info["name"]
            categorized.append(tx)
        
        opportunities = categorizer_agent._analyze_tax_optimization(categorized)
        
        # Should identify meal deduction opportunity (restaurant transaction)
        meal_opportunity = next((opp for opp in opportunities if opp["type"] == "meal_deduction_optimization"), None)
        assert meal_opportunity is not None
        
        # Should identify equipment Section 179 opportunity (laptop purchase)
        equipment_opportunity = next((opp for opp in opportunities if opp["type"] == "section_179_deduction"), None)
        assert equipment_opportunity is not None
        
        # Should identify large transaction review (equipment purchase)
        large_tx_opportunity = next((opp for opp in opportunities if opp["type"] == "large_transaction_review"), None)
        assert large_tx_opportunity is not None
    
    def test_compliance_risk_assessment_realistic(self, categorizer_agent, real_world_transactions):
        """Test compliance risk assessment with realistic scenarios."""
        # Categorize transactions first
        categorized = []
        for tx in real_world_transactions:
            category_info = categorizer_agent._determine_category(tx)
            tx.tax_category = category_info["name"]
            categorized.append(tx)
        
        risks = categorizer_agent._assess_compliance_risks(categorized)
        
        # Should flag Form 8300 requirement for $15,000 transaction
        form_8300_risk = next((risk for risk in risks if risk["type"] == "form_8300_requirement"), None)
        assert form_8300_risk is not None
        assert form_8300_risk["severity"] == "high"
        
        # Should flag uncategorized transactions
        uncat_risk = next((risk for risk in risks if risk["type"] == "uncategorized_transactions"), None)
        assert uncat_risk is not None
        assert len(uncat_risk["transactions"]) >= 1  # At least the misc expense
    
    def test_tax_metrics_calculation_realistic(self, categorizer_agent, real_world_transactions):
        """Test tax metrics calculation with realistic amounts."""
        # Categorize transactions first
        categorized = []
        for tx in real_world_transactions:
            category_info = categorizer_agent._determine_category(tx)
            tx.tax_category = category_info["name"]
            categorized.append(tx)
        
        metrics = categorizer_agent._calculate_tax_metrics(categorized)
        
        # Verify total expenses calculation
        expected_total = sum(tx.amount for tx in real_world_transactions)
        assert metrics["total_expenses"] == expected_total
        
        # Verify deductible amount is less than total (due to 50% meal deduction and uncategorized)
        assert metrics["total_deductible"] < metrics["total_expenses"]
        assert metrics["total_deductible"] > Decimal("0")
        
        # Verify estimated savings
        assert metrics["estimated_savings"] > Decimal("0")
        assert metrics["estimated_savings"] == metrics["total_deductible"] * Decimal("0.25")
        
        # Verify deduction percentage is reasonable
        assert 0 < metrics["deduction_percentage"] < 1
    
    def test_citation_generation_comprehensive(self, categorizer_agent, real_world_transactions):
        """Test that citations are generated for all categorized transactions."""
        # Run categorization to generate citations
        categorized = []
        for tx in real_world_transactions:
            category_info = categorizer_agent._determine_category(tx)
            tx.tax_category = category_info["name"]
            categorized.append(tx)
            
            # Simulate citation generation (normally done in _categorize_transactions)
            categorizer_agent.citations.append({
                "transaction_id": tx.id,
                "category": category_info["name"],
                "irs_reference": category_info["irs_reference"],
                "deduction_rate": float(category_info["deduction_rate"]),
                "reasoning": f"Categorized as {category_info['name']} based on description analysis"
            })
        
        citations = categorizer_agent._generate_citation_report()
        
        # Should have citations for all transactions
        assert len(citations) == len(real_world_transactions)
        
        # Verify citation structure
        for citation in citations:
            assert "transaction_id" in citation
            assert "category" in citation
            assert "irs_reference" in citation
            assert "deduction_rate" in citation
            assert "reasoning" in citation
            
            # Verify IRS reference is provided
            assert citation["irs_reference"] != ""
            assert "IRS" in citation["irs_reference"] or "Form" in citation["irs_reference"] or "Manual" in citation["irs_reference"]
    
    @pytest.mark.asyncio
    async def test_error_handling_malformed_data(self, categorizer_agent):
        """Test error handling with malformed transaction data."""
        malformed_state = WorkflowState(
            workflow_id="error-test-workflow",
            analysis=AnalysisState(
                transactions=[
                    {
                        "id": "bad_tx",
                        "date": "invalid_date",  # This will cause an error
                        "amount": "not_a_number",
                        "description": "Test transaction",
                        "source_file": "test.csv",
                        "data_quality_score": 0.5,
                        "data_issues": []
                    }
                ],
                status=AgentStatus.COMPLETED
            )
        )
        
        with pytest.raises(Exception):  # Should raise validation error
            await categorizer_agent.execute(malformed_state)
    
    def test_memory_updates(self, categorizer_agent, real_world_transactions):
        """Test that agent memory is properly updated during execution."""
        # Simulate execution by calling internal methods
        categorized = []
        for tx in real_world_transactions:
            category_info = categorizer_agent._determine_category(tx)
            tx.tax_category = category_info["name"]
            categorized.append(tx)
        
        metrics = categorizer_agent._calculate_tax_metrics(categorized)
        optimization_opportunities = categorizer_agent._analyze_tax_optimization(categorized)
        compliance_risks = categorizer_agent._assess_compliance_risks(categorized)
        
        # Update memory as would happen in execute()
        categorizer_agent.update_memory("categorized_transactions", len(categorized))
        categorizer_agent.update_memory("total_deductible", float(metrics["total_deductible"]))
        categorizer_agent.update_memory("optimization_opportunities", len(optimization_opportunities))
        categorizer_agent.update_memory("compliance_risks", len(compliance_risks))
        
        # Verify memory contents
        assert categorizer_agent.get_memory("categorized_transactions") == len(categorized)
        assert categorizer_agent.get_memory("total_deductible") == float(metrics["total_deductible"])
        assert categorizer_agent.get_memory("optimization_opportunities") == len(optimization_opportunities)
        assert categorizer_agent.get_memory("compliance_risks") == len(compliance_risks)
    
    def test_conservative_compliance_approach(self, categorizer_agent):
        """Test that the agent takes a conservative approach to compliance."""
        # Test with ambiguous transaction
        ambiguous_tx = Transaction(
            id="ambiguous",
            date=datetime.now(),
            amount=Decimal("500.00"),
            description="Payment to vendor for services",  # Could be many categories
            source_file="test.csv",
            data_quality_score=0.7,
            data_issues=["Ambiguous description"]
        )
        
        category_info = categorizer_agent._determine_category(ambiguous_tx)
        
        # Should default to uncategorized for conservative compliance
        assert category_info["name"] == "Uncategorized - Requires Review"
        assert category_info["deduction_rate"] == Decimal("0.0")
    
    def test_audit_defensibility(self, categorizer_agent, real_world_transactions):
        """Test that all categorization decisions are audit-defensible."""
        categorized = []
        for tx in real_world_transactions:
            category_info = categorizer_agent._determine_category(tx)
            tx.tax_category = category_info["name"]
            categorized.append(tx)
            
            # Verify each category has proper IRS reference
            assert category_info["irs_reference"] is not None
            assert category_info["irs_reference"] != ""
            
            # Verify deduction rate is reasonable
            assert Decimal("0.0") <= category_info["deduction_rate"] <= Decimal("1.0")
        
        # Verify compliance risks are identified
        risks = categorizer_agent._assess_compliance_risks(categorized)
        
        # Should identify at least some risks (large transactions, etc.)
        assert len(risks) > 0
        
        # Each risk should have proper documentation
        for risk in risks:
            assert "type" in risk
            assert "severity" in risk
            assert "description" in risk
            assert "action" in risk
            assert risk["severity"] in ["low", "medium", "high"]