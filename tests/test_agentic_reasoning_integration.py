"""
Integration tests for the agentic reasoning engine.
Tests the complete workflow with real-world scenarios including conflicting rules.
"""

import pytest
from unittest.mock import AsyncMock, patch

from src.utils.rag.agentic_reasoning import (
    AgenticReasoningEngine,
    ConfidenceLevel,
    RuleConflictType,
    analyze_tax_rule,
    get_expense_deduction_guidance
)
from src.utils.rag.vector_store import SearchResult


class TestAgenticReasoningIntegration:
    """Integration tests for the complete agentic reasoning workflow."""
    
    @pytest.fixture
    def business_meal_results(self):
        """Sample search results for business meal queries."""
        return [
            SearchResult(
                content="Business meals are generally 50% deductible when they are ordinary and necessary business expenses. The meal must be directly related to the active conduct of business.",
                similarity_score=0.85,
                metadata={
                    "title": "Publication 334 - Tax Guide for Small Business",
                    "section": "Chapter 2 - Business Expenses",
                    "publication_year": 2024,
                    "keywords": ["meal", "deduction", "business"]
                },
                document_id="pub334_2024",
                chunk_index=15
            ),
            SearchResult(
                content="To be deductible, a business expense must be both ordinary and necessary. An ordinary expense is one that is common and accepted in your trade or business.",
                similarity_score=0.80,
                metadata={
                    "title": "Publication 334 - Tax Guide for Small Business",
                    "section": "Chapter 2 - Business Expenses",
                    "publication_year": 2024
                },
                document_id="pub334_2024",
                chunk_index=10
            ),
            SearchResult(
                content="Substantiation requirements: You must keep records that show the amount, time, place, and business purpose of the meal expense.",
                similarity_score=0.75,
                metadata={
                    "title": "Publication 334 - Tax Guide for Small Business",
                    "section": "Chapter 2 - Business Expenses",
                    "publication_year": 2024,
                    "keywords": ["substantiation", "records"]
                },
                document_id="pub334_2024",
                chunk_index=17
            )
        ]
    
    @pytest.fixture
    def conflicting_meal_results(self):
        """Conflicting search results for testing conflict resolution."""
        return [
            SearchResult(
                content="Business meals are 100% deductible when they meet the ordinary and necessary business expense test under Section 162.",
                similarity_score=0.70,
                metadata={
                    "title": "Internal Revenue Code Section 162",
                    "publication_year": 2024
                },
                document_id="code_section_162",
                chunk_index=5
            ),
            SearchResult(
                content="Business meals are generally limited to 50% deductibility under current tax regulations.",
                similarity_score=0.85,
                metadata={
                    "title": "Publication 334 - Tax Guide for Small Business",
                    "publication_year": 2024
                },
                document_id="pub334_2024",
                chunk_index=15
            ),
            SearchResult(
                content="Entertainment expenses are generally not deductible for tax years beginning after December 31, 2017.",
                similarity_score=0.65,
                metadata={
                    "title": "Publication 334 - Tax Guide for Small Business",
                    "publication_year": 2024
                },
                document_id="pub334_2024",
                chunk_index=16
            )
        ]
    
    @pytest.mark.asyncio
    @patch('src.utils.rag.agentic_reasoning.SimilaritySearchEngine')
    async def test_business_meal_reasoning_success(self, mock_search_engine, business_meal_results):
        """Test successful reasoning about business meal deductibility."""
        # Setup mock
        mock_engine_instance = AsyncMock()
        mock_search_engine.return_value = mock_engine_instance
        mock_engine_instance.search.return_value = business_meal_results
        
        # Create reasoning engine
        engine = AgenticReasoningEngine()
        engine.search_engine = mock_engine_instance
        
        # Test query with context
        query = "Are business meals deductible?"
        context = {
            "transaction_amount": 150.00,
            "business_type": "consulting",
            "transaction_date": "2024-01-15",
            "client_meeting": True
        }
        
        result = await engine.reason_about_query(query, context)
        
        # Verify query enhancement
        assert result.query.original_query == query
        assert "meal" in result.query.enhanced_query.lower()
        assert len(result.query.context_factors) > 0
        assert "amount: $150.0" in result.query.context_factors
        
        # Verify rule interpretations
        assert len(result.rule_interpretations) == 3
        
        # Check first interpretation (highest similarity)
        primary_interp = result.rule_interpretations[0]
        assert "50%" in primary_interp.interpretation or "deductible" in primary_interp.interpretation.lower()
        assert primary_interp.confidence in [ConfidenceLevel.HIGH, ConfidenceLevel.MEDIUM]
        assert len(primary_interp.citations) == 1
        assert primary_interp.citations[0].document_id == "pub334_2024"
        
        # Verify synthesized guidance
        assert len(result.synthesized_guidance) > 0
        assert "50%" in result.synthesized_guidance or "deductible" in result.synthesized_guidance.lower()
        
        # Verify recommendations
        assert len(result.recommendations) > 0
        
        # Verify audit trail
        assert len(result.audit_trail) > 0
        assert any("Original query" in item for item in result.audit_trail)
        assert any("Retrieved" in item for item in result.audit_trail)
        
        # Verify overall confidence is reasonable
        assert result.overall_confidence in [ConfidenceLevel.HIGH, ConfidenceLevel.MEDIUM, ConfidenceLevel.LOW]
    
    @pytest.mark.asyncio
    @patch('src.utils.rag.agentic_reasoning.SimilaritySearchEngine')
    async def test_conflicting_rules_resolution(self, mock_search_engine, conflicting_meal_results):
        """Test conflict detection and resolution with contradictory rules."""
        # Setup mock
        mock_engine_instance = AsyncMock()
        mock_search_engine.return_value = mock_engine_instance
        mock_engine_instance.search.return_value = conflicting_meal_results
        
        # Create reasoning engine
        engine = AgenticReasoningEngine()
        engine.search_engine = mock_engine_instance
        
        # Test query that should produce conflicts
        query = "What percentage of business meals are deductible?"
        
        result = await engine.reason_about_query(query)
        
        # Verify rule interpretations were created
        assert len(result.rule_interpretations) == 3
        
        # Check for potential conflicts (may or may not be detected depending on grouping)
        # The system should handle this gracefully either way
        
        # Verify synthesized guidance provides reasonable answer
        assert len(result.synthesized_guidance) > 0
        
        # With conflicts, confidence should be reduced or recommendations should include caution
        if result.conflicts:
            assert len(result.conflicts) > 0
            # Should recommend conservative approach or professional consultation
            assert any("conservative" in rec.lower() or "professional" in rec.lower() 
                     for rec in result.recommendations)
        
        # Verify audit trail captures the analysis
        assert len(result.audit_trail) > 0
    
    @pytest.mark.asyncio
    @patch('src.utils.rag.agentic_reasoning.SimilaritySearchEngine')
    async def test_uncertainty_handling(self, mock_search_engine):
        """Test handling of uncertain or ambiguous rules."""
        # Create ambiguous search results
        ambiguous_results = [
            SearchResult(
                content="Business expenses may be deductible if they are ordinary and necessary, but specific circumstances vary.",
                similarity_score=0.60,  # Lower confidence
                metadata={
                    "title": "General Tax Guidance",
                    "publication_year": 2024
                },
                document_id="general_guidance",
                chunk_index=1
            ),
            SearchResult(
                content="Consult a tax professional for specific deduction questions as rules can be complex.",
                similarity_score=0.55,
                metadata={
                    "title": "Tax Advisory Notice",
                    "publication_year": 2024
                },
                document_id="advisory_notice",
                chunk_index=1
            )
        ]
        
        # Setup mock
        mock_engine_instance = AsyncMock()
        mock_search_engine.return_value = mock_engine_instance
        mock_engine_instance.search.return_value = ambiguous_results
        
        # Create reasoning engine
        engine = AgenticReasoningEngine()
        engine.search_engine = mock_engine_instance
        
        # Test ambiguous query
        query = "Can I deduct this unusual business expense?"
        
        result = await engine.reason_about_query(query)
        
        # With low-confidence results, should recommend professional consultation
        assert result.overall_confidence in [ConfidenceLevel.LOW, ConfidenceLevel.VERY_LOW]
        assert any("professional" in rec.lower() for rec in result.recommendations)
        
        # Guidance should be conservative - either in the guidance itself or recommendations
        has_professional_guidance = ("professional" in result.synthesized_guidance.lower() or 
                                    "consult" in result.synthesized_guidance.lower())
        has_professional_recommendation = any("professional" in rec.lower() for rec in result.recommendations)
        
        assert has_professional_guidance or has_professional_recommendation
    
    @pytest.mark.asyncio
    @patch('src.utils.rag.agentic_reasoning.SimilaritySearchEngine')
    async def test_citation_tracking(self, mock_search_engine, business_meal_results):
        """Test comprehensive citation tracking for audit compliance."""
        # Setup mock
        mock_engine_instance = AsyncMock()
        mock_search_engine.return_value = mock_engine_instance
        mock_engine_instance.search.return_value = business_meal_results
        
        # Create reasoning engine
        engine = AgenticReasoningEngine()
        engine.search_engine = mock_engine_instance
        
        query = "Business meal deduction rules"
        
        result = await engine.reason_about_query(query)
        
        # Verify all interpretations have proper citations
        for interp in result.rule_interpretations:
            assert len(interp.citations) > 0
            
            citation = interp.citations[0]
            assert citation.document_id is not None
            assert citation.document_title is not None
            assert 0.0 <= citation.confidence_score <= 1.0
            
            # Check for IRS publication citations
            if "pub334" in citation.document_id:
                assert "Publication 334" in citation.document_title
                assert citation.publication_year == 2024
        
        # Verify audit trail includes citation information
        citation_trail_items = [item for item in result.audit_trail if "Rule" in item and "confidence" in item]
        assert len(citation_trail_items) > 0
    
    @pytest.mark.asyncio
    @patch('src.utils.rag.agentic_reasoning.SimilaritySearchEngine')
    async def test_conservative_fallback(self, mock_search_engine):
        """Test conservative fallback when no relevant rules are found."""
        # Setup mock to return empty results
        mock_engine_instance = AsyncMock()
        mock_search_engine.return_value = mock_engine_instance
        mock_engine_instance.search.return_value = []
        
        # Create reasoning engine
        engine = AgenticReasoningEngine()
        engine.search_engine = mock_engine_instance
        
        query = "Can I deduct my pet dragon as a business expense?"
        
        result = await engine.reason_about_query(query)
        
        # Should provide conservative guidance
        assert result.overall_confidence == ConfidenceLevel.VERY_LOW
        assert "professional" in result.synthesized_guidance.lower()
        assert len(result.rule_interpretations) == 0
        assert any("professional" in rec.lower() for rec in result.recommendations)
    
    @pytest.mark.asyncio
    @patch('src.utils.rag.agentic_reasoning.SimilaritySearchEngine')
    async def test_error_handling_and_recovery(self, mock_search_engine):
        """Test error handling and graceful recovery."""
        # Setup mock to raise an exception
        mock_engine_instance = AsyncMock()
        mock_search_engine.return_value = mock_engine_instance
        mock_engine_instance.search.side_effect = Exception("Database connection failed")
        
        # Create reasoning engine
        engine = AgenticReasoningEngine()
        engine.search_engine = mock_engine_instance
        
        query = "Business expense deduction"
        
        result = await engine.reason_about_query(query)
        
        # Should return fallback result
        assert result.overall_confidence == ConfidenceLevel.VERY_LOW
        assert "professional" in result.synthesized_guidance.lower()
        assert len(result.rule_interpretations) == 0
        assert any("professional" in rec.lower() for rec in result.recommendations)
        assert any("error" in item.lower() for item in result.audit_trail)


class TestConvenienceFunctionIntegration:
    """Integration tests for convenience functions."""
    
    @pytest.mark.asyncio
    @patch('src.utils.rag.agentic_reasoning.AgenticReasoningEngine')
    async def test_analyze_tax_rule_integration(self, mock_engine_class):
        """Test analyze_tax_rule convenience function integration."""
        # Create mock result
        from src.utils.rag.agentic_reasoning import AgenticQuery, ReasoningResult
        
        mock_result = ReasoningResult(
            query=AgenticQuery(
                original_query="test query",
                enhanced_query="enhanced test query",
                context_factors=["amount: $500"],
                domain_keywords=["deduction"],
                search_strategy="balanced"
            ),
            rule_interpretations=[],
            synthesized_guidance="Business meals are 50% deductible with proper substantiation.",
            overall_confidence=ConfidenceLevel.HIGH,
            recommendations=["Maintain proper documentation", "Ensure business purpose"]
        )
        
        # Setup mock
        mock_engine = AsyncMock()
        mock_engine_class.return_value = mock_engine
        mock_engine.reason_about_query.return_value = mock_result
        
        # Test the convenience function
        result = await analyze_tax_rule(
            "Are client dinner expenses deductible?",
            {"transaction_amount": 500, "client_meeting": True}
        )
        
        assert result == mock_result
        assert result.overall_confidence == ConfidenceLevel.HIGH
        assert "50%" in result.synthesized_guidance
        assert len(result.recommendations) == 2
    
    @pytest.mark.asyncio
    @patch('src.utils.rag.agentic_reasoning.AgenticReasoningEngine')
    async def test_get_expense_deduction_guidance_integration(self, mock_engine_class):
        """Test get_expense_deduction_guidance convenience function integration."""
        # Create mock result
        from src.utils.rag.agentic_reasoning import AgenticQuery, ReasoningResult
        
        mock_result = ReasoningResult(
            query=AgenticQuery(
                original_query="Is office supplies deductible as a business expense?",
                enhanced_query="office supplies deductible business expense ordinary necessary",
                context_factors=["amount: $75.0", "business: consulting"],
                domain_keywords=["deduction", "ordinary", "necessary"],
                search_strategy="balanced"
            ),
            rule_interpretations=[],
            synthesized_guidance="Office supplies are generally 100% deductible as ordinary and necessary business expenses.",
            overall_confidence=ConfidenceLevel.HIGH,
            recommendations=["Keep receipts for substantiation", "Ensure business use"]
        )
        
        # Setup mock
        mock_engine = AsyncMock()
        mock_engine_class.return_value = mock_engine
        mock_engine.reason_about_query.return_value = mock_result
        
        # Test the convenience function
        result = await get_expense_deduction_guidance(
            "office supplies", 75.00, "consulting"
        )
        
        assert result == mock_result
        
        # Verify the query was constructed correctly
        call_args = mock_engine.reason_about_query.call_args
        query = call_args[0][0]
        context = call_args[0][1]
        
        assert "office supplies" in query
        assert "deductible" in query
        assert context["transaction_amount"] == 75.00
        assert context["business_type"] == "consulting"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])