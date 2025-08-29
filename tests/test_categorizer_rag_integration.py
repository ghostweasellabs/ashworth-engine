"""
Integration tests for the Categorizer Agent with agentic RAG system.
Tests the complete workflow with RAG-enhanced categorization and fallback behavior.
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from decimal import Decimal
from datetime import datetime

from src.agents.categorizer import CategorizerAgent
from src.models.base import Transaction
from src.workflows.state_schemas import WorkflowState, AnalysisState
from src.utils.rag.agentic_reasoning import (
    ReasoningResult, 
    AgenticQuery, 
    RuleInterpretation, 
    Citation, 
    ConfidenceLevel
)


class TestCategorizerRAGIntegration:
    """Integration tests for RAG-enhanced categorizer agent."""
    
    @pytest.fixture
    def sample_transactions(self):
        """Sample transactions for testing."""
        return [
            Transaction(
                id="tx_001",
                date=datetime(2024, 1, 15),
                amount=Decimal("150.00"),
                description="Business lunch with client",
                counterparty="Restaurant ABC",
                source_file="test.csv",
                data_quality_score=0.9
            ),
            Transaction(
                id="tx_002", 
                date=datetime(2024, 1, 16),
                amount=Decimal("2500.00"),
                description="Office computer purchase",
                counterparty="Tech Store",
                source_file="test.csv",
                data_quality_score=0.95
            ),
            Transaction(
                id="tx_003",
                date=datetime(2024, 1, 17),
                amount=Decimal("75.00"),
                description="Office supplies",
                counterparty="Staples",
                source_file="test.csv",
                data_quality_score=0.85
            )
        ]
    
    @pytest.fixture
    def high_confidence_rag_result(self):
        """High confidence RAG result for business meal."""
        return ReasoningResult(
            query=AgenticQuery(
                original_query="Business lunch with client",
                enhanced_query="business lunch client meal deduction",
                context_factors=["amount: $150.0"],
                domain_keywords=["meal", "deduction"],
                search_strategy="balanced"
            ),
            rule_interpretations=[
                RuleInterpretation(
                    rule_text="Business meals are generally 50% deductible when they are ordinary and necessary business expenses.",
                    interpretation="This expense may be deductible. Subject to 50% limitation.",
                    confidence=ConfidenceLevel.HIGH,
                    reasoning="Based on 85% similarity to query. Found in IRS Publication 334.",
                    citations=[
                        Citation(
                            document_id="pub334_2024",
                            document_title="Publication 334 - Tax Guide for Small Business",
                            section="Chapter 11",
                            confidence_score=0.85,
                            publication_year=2024
                        )
                    ]
                )
            ],
            synthesized_guidance="Business meals are 50% deductible with proper substantiation and business purpose documentation.",
            overall_confidence=ConfidenceLevel.HIGH,
            recommendations=["Maintain proper documentation", "Ensure business purpose"]
        )
    
    @pytest.fixture
    def low_confidence_rag_result(self):
        """Low confidence RAG result that should trigger fallback."""
        return ReasoningResult(
            query=AgenticQuery(
                original_query="Unusual business expense",
                enhanced_query="unusual business expense deduction",
                context_factors=["amount: $500.0"],
                domain_keywords=["deduction"],
                search_strategy="balanced"
            ),
            rule_interpretations=[
                RuleInterpretation(
                    rule_text="Business expenses may be deductible if they are ordinary and necessary.",
                    interpretation="Deductibility depends on specific circumstances.",
                    confidence=ConfidenceLevel.LOW,
                    reasoning="Based on 45% similarity to query. Ambiguous guidance.",
                    citations=[
                        Citation(
                            document_id="general_guidance",
                            document_title="General Tax Guidance",
                            confidence_score=0.45
                        )
                    ]
                )
            ],
            synthesized_guidance="Consult a tax professional for specific deduction questions as rules can be complex.",
            overall_confidence=ConfidenceLevel.LOW,
            recommendations=["Professional consultation recommended"]
        )
    
    @pytest.mark.asyncio
    @patch('src.agents.categorizer.get_expense_deduction_guidance')
    async def test_high_confidence_rag_categorization(self, mock_rag_guidance, sample_transactions, high_confidence_rag_result):
        """Test successful RAG categorization with high confidence."""
        # Setup mock
        mock_rag_guidance.return_value = high_confidence_rag_result
        
        # Create agent and state
        agent = CategorizerAgent()
        state = WorkflowState(
            workflow_id="test_workflow",
            analysis=AnalysisState(
                transactions=[tx.model_dump() for tx in sample_transactions[:1]]  # Just the meal transaction
            )
        )
        
        # Execute categorization
        result_state = await agent.execute(state)
        
        # Verify RAG was called
        mock_rag_guidance.assert_called_once()
        call_args = mock_rag_guidance.call_args[0]
        assert "Business lunch with client" in call_args[0]
        assert call_args[1] == 150.0
        
        # Verify categorization result
        analysis = result_state["analysis"]
        categorized_transactions = analysis["transactions"]
        
        assert len(categorized_transactions) == 1
        tx = categorized_transactions[0]
        assert tx["tax_category"] == "Meals and Entertainment"
        
        # Verify tax implications include RAG insights
        tax_implications = analysis["tax_implications"]
        assert "citations" in tax_implications
        
        citations = tax_implications["citations"]
        assert len(citations) == 1
        
        citation = citations[0]
        assert citation["confidence"] == ConfidenceLevel.HIGH.value
        assert citation["rag_citations"] is not None
        assert len(citation["rag_citations"]) > 0
        assert not citation["fallback_used"]
        
        # Verify RAG citation details
        rag_citation = citation["rag_citations"][0]
        assert rag_citation["document_title"] == "Publication 334 - Tax Guide for Small Business"
        assert rag_citation["section"] == "Chapter 11"
        assert rag_citation["confidence_score"] == 0.85
    
    @pytest.mark.asyncio
    @patch('src.agents.categorizer.get_expense_deduction_guidance')
    async def test_low_confidence_fallback_categorization(self, mock_rag_guidance, sample_transactions, low_confidence_rag_result):
        """Test fallback to conservative categorization when RAG confidence is low."""
        # Setup mock
        mock_rag_guidance.return_value = low_confidence_rag_result
        
        # Create agent and state with office supplies transaction (should match fallback rules)
        agent = CategorizerAgent()
        state = WorkflowState(
            workflow_id="test_workflow",
            analysis=AnalysisState(
                transactions=[sample_transactions[2].model_dump()]  # Office supplies transaction
            )
        )
        
        # Execute categorization
        result_state = await agent.execute(state)
        
        # Verify RAG was called
        mock_rag_guidance.assert_called_once()
        
        # Verify fallback categorization was used
        analysis = result_state["analysis"]
        categorized_transactions = analysis["transactions"]
        
        assert len(categorized_transactions) == 1
        tx = categorized_transactions[0]
        # Should use fallback rule-based categorization for office supplies
        assert tx["tax_category"] == "Office Expenses"
        
        # Verify citation shows fallback was used
        citations = analysis["tax_implications"]["citations"]
        assert len(citations) == 1
        
        citation = citations[0]
        assert citation["confidence"] == ConfidenceLevel.LOW.value
        assert citation["fallback_used"] is True
        assert "fallback" in citation["reasoning"].lower()
        
        # Should still include RAG citations for audit trail
        assert citation["rag_citations"] is not None
    
    @pytest.mark.asyncio
    @patch('src.agents.categorizer.get_expense_deduction_guidance')
    async def test_rag_error_handling(self, mock_rag_guidance, sample_transactions):
        """Test error handling when RAG system fails."""
        # Setup mock to raise exception
        mock_rag_guidance.side_effect = Exception("RAG system unavailable")
        
        # Create agent and state
        agent = CategorizerAgent()
        state = WorkflowState(
            workflow_id="test_workflow",
            analysis=AnalysisState(
                transactions=[sample_transactions[0].model_dump()]
            )
        )
        
        # Execute categorization - should not raise exception
        result_state = await agent.execute(state)
        
        # Verify fallback categorization was used
        analysis = result_state["analysis"]
        categorized_transactions = analysis["transactions"]
        
        assert len(categorized_transactions) == 1
        tx = categorized_transactions[0]
        # Should use conservative fallback
        assert tx["tax_category"] is not None
        
        # Verify error is tracked in citation
        citations = analysis["tax_implications"]["citations"]
        assert len(citations) == 1
        
        citation = citations[0]
        assert citation["fallback_used"] is True
        assert "error" in citation
        assert "RAG system unavailable" in citation["error"]
    
    @pytest.mark.asyncio
    @patch('src.agents.categorizer.get_expense_deduction_guidance')
    async def test_rag_enhanced_optimization_analysis(self, mock_rag_guidance, sample_transactions, high_confidence_rag_result):
        """Test that optimization analysis includes RAG insights."""
        # Setup mock
        mock_rag_guidance.return_value = high_confidence_rag_result
        
        # Create agent and state with meal transaction
        agent = CategorizerAgent()
        state = WorkflowState(
            workflow_id="test_workflow",
            analysis=AnalysisState(
                transactions=[sample_transactions[0].model_dump()]  # Business meal
            )
        )
        
        # Execute categorization
        result_state = await agent.execute(state)
        
        # Verify optimization opportunities include RAG enhancement
        tax_implications = result_state["analysis"]["tax_implications"]
        optimization_opportunities = tax_implications["optimization_opportunities"]
        
        # Should have meal deduction optimization
        meal_optimization = next(
            (opp for opp in optimization_opportunities if opp["type"] == "meal_deduction_optimization"),
            None
        )
        
        assert meal_optimization is not None
        assert meal_optimization["rag_enhanced"] is True
        assert "Publication 334" in meal_optimization["irs_reference"]
    
    @pytest.mark.asyncio
    @patch('src.agents.categorizer.get_expense_deduction_guidance')
    async def test_rag_enhanced_compliance_analysis(self, mock_rag_guidance, sample_transactions, high_confidence_rag_result, low_confidence_rag_result):
        """Test that compliance analysis includes RAG-specific risks."""
        # Setup mock to return mixed confidence results
        mock_rag_guidance.side_effect = [
            high_confidence_rag_result,
            low_confidence_rag_result,
            low_confidence_rag_result
        ]
        
        # Create agent and state with multiple transactions
        agent = CategorizerAgent()
        state = WorkflowState(
            workflow_id="test_workflow",
            analysis=AnalysisState(
                transactions=[tx.model_dump() for tx in sample_transactions]
            )
        )
        
        # Execute categorization
        result_state = await agent.execute(state)
        
        # Verify compliance risks include RAG-specific analysis
        compliance_risks = result_state["analysis"]["tax_implications"]["compliance_risks"]
        
        # Should identify low confidence categorizations as a risk
        low_confidence_risk = next(
            (risk for risk in compliance_risks if risk["type"] == "low_confidence_categorizations"),
            None
        )
        
        assert low_confidence_risk is not None
        assert low_confidence_risk["rag_enhanced"] is True
        assert low_confidence_risk["transaction_count"] > 0
    
    @pytest.mark.asyncio
    @patch('src.agents.categorizer.get_expense_deduction_guidance')
    async def test_citation_report_enhancement(self, mock_rag_guidance, sample_transactions, high_confidence_rag_result):
        """Test that citation reports include comprehensive RAG analysis."""
        # Setup mock
        mock_rag_guidance.return_value = high_confidence_rag_result
        
        # Create agent and state
        agent = CategorizerAgent()
        state = WorkflowState(
            workflow_id="test_workflow",
            analysis=AnalysisState(
                transactions=[sample_transactions[0].model_dump()]
            )
        )
        
        # Execute categorization
        result_state = await agent.execute(state)
        
        # Verify enhanced citation report
        citations = result_state["analysis"]["tax_implications"]["citations"]
        assert len(citations) == 1
        
        citation = citations[0]
        
        # Should have RAG analysis summary
        assert "rag_analysis" in citation
        rag_analysis = citation["rag_analysis"]
        assert rag_analysis["rag_sources_count"] > 0
        assert rag_analysis["highest_confidence_source"] is not None
        assert len(rag_analysis["document_sources"]) > 0
        
        # Should have compliance flags
        assert "compliance_flags" in citation
        compliance_flags = citation["compliance_flags"]
        assert compliance_flags["high_confidence"] is True
        assert compliance_flags["rag_enhanced"] is True
        assert compliance_flags["fallback_used"] is False
        assert compliance_flags["professional_review_recommended"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])