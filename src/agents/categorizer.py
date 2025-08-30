"""
Categorizer Agent - Clarke Pemberton, JD, CPA
Corporate Tax Compliance Strategist with strategic tax optimization mindset.
Enhanced with agentic RAG for IRS compliance and rule-based categorization.
"""

import asyncio
import logging
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple, Set
import re

from src.agents.base import BaseAgent
from src.config.personas import CATEGORIZER_PERSONALITY
from src.workflows.state_schemas import WorkflowState, AgentStatus, AnalysisState
from src.models.base import Transaction
from src.utils.rag.agentic_reasoning import (
    AgenticReasoningEngine, 
    ConfidenceLevel, 
    get_expense_deduction_guidance,
    analyze_tax_rule
)


class CategorizerAgent(BaseAgent):
    """
    Clarke Pemberton, JD, CPA - Corporate Tax Compliance Strategist
    
    Persona: Big Four accounting firm veteran with JD and CPA credentials,
    specializing in strategic tax optimization and IRS compliance with
    meticulous attention to audit defensibility.
    """
    
    def __init__(self):
        super().__init__(CATEGORIZER_PERSONALITY)
        
        # Initialize agentic RAG engine for IRS rule retrieval
        self.rag_engine = AgenticReasoningEngine()
        
        # Fallback IRS expense categories for when RAG confidence is low
        self.fallback_categories = {
            "utilities": {
                "name": "Utilities",
                "deduction_rate": Decimal("1.0"),
                "keywords": ["electric", "electricity", "gas", "water", "internet", "phone", "telephone", "utility", "power", "energy"],
                "priority_keywords": ["electric", "electricity", "gas", "water", "utility"],
                "irs_reference": "IRS Publication 334, Chapter 8"
            },
            "office_expenses": {
                "name": "Office Expenses",
                "deduction_rate": Decimal("1.0"),
                "keywords": ["supplies", "stationery", "paper", "ink", "printer", "desk", "chair", "staples"],
                "priority_keywords": ["supplies", "stationery", "staples"],
                "irs_reference": "IRS Publication 334, Chapter 8"
            },
            "travel": {
                "name": "Travel Expenses", 
                "deduction_rate": Decimal("1.0"),
                "keywords": ["hotel", "flight", "airline", "taxi", "uber", "lyft", "rental car", "mileage", "travel"],
                "priority_keywords": ["flight", "airline", "hotel", "travel"],
                "irs_reference": "IRS Publication 334, Chapter 11"
            },
            "meals": {
                "name": "Meals and Entertainment",
                "deduction_rate": Decimal("0.5"),  # 50% deduction for business meals
                "keywords": ["restaurant", "meal", "lunch", "dinner", "catering", "food"],
                "priority_keywords": ["restaurant", "meal", "lunch", "dinner"],
                "irs_reference": "IRS Publication 334, Chapter 11"
            },
            "professional_services": {
                "name": "Professional Services",
                "deduction_rate": Decimal("1.0"),
                "keywords": ["legal", "attorney", "accounting", "consultant", "professional", "advisory"],
                "priority_keywords": ["legal", "attorney", "accounting", "consultant"],
                "irs_reference": "IRS Publication 334, Chapter 8"
            },
            "equipment": {
                "name": "Equipment and Depreciation",
                "deduction_rate": Decimal("1.0"),
                "keywords": ["computer", "laptop", "software", "equipment", "machinery", "tools"],
                "priority_keywords": ["computer", "laptop", "equipment", "machinery"],
                "irs_reference": "IRS Publication 334, Chapter 9"
            },
            "marketing": {
                "name": "Marketing and Advertising",
                "deduction_rate": Decimal("1.0"),
                "keywords": ["advertising", "marketing", "promotion", "website", "social media", "branding", "ads"],
                "priority_keywords": ["advertising", "marketing", "ads"],
                "irs_reference": "IRS Publication 334, Chapter 8"
            },
            "insurance": {
                "name": "Insurance",
                "deduction_rate": Decimal("1.0"),
                "keywords": ["insurance", "premium", "liability", "coverage", "policy"],
                "priority_keywords": ["insurance", "premium", "liability"],
                "irs_reference": "IRS Publication 334, Chapter 8"
            },
            "rent": {
                "name": "Rent and Lease",
                "deduction_rate": Decimal("1.0"),
                "keywords": ["rent", "lease", "rental", "office space", "warehouse"],
                "priority_keywords": ["rent", "lease", "rental"],
                "irs_reference": "IRS Publication 334, Chapter 8"
            },
            "uncategorized": {
                "name": "Uncategorized - Requires Review",
                "deduction_rate": Decimal("0.0"),  # Conservative approach
                "keywords": [],
                "priority_keywords": [],
                "irs_reference": "Manual review required"
            }
        }
        
        # Tax optimization thresholds and rules
        self.optimization_rules = {
            "large_transaction_threshold": Decimal("10000.00"),  # Form 8300 reporting requirement
            "meal_deduction_limit": Decimal("0.5"),  # 50% business meal deduction
            "home_office_threshold": Decimal("500.00"),  # Simplified home office deduction
            "vehicle_expense_threshold": Decimal("5000.00"),  # Consider actual vs standard mileage
        }
        
        # Compliance flags and audit triggers
        self.audit_triggers = {
            "cash_transaction_limit": Decimal("10000.00"),  # Form 8300 requirement
            "round_number_percentage": 0.8,  # High percentage of round numbers
            "entertainment_percentage": 0.3,  # High entertainment expenses
            "home_office_percentage": 0.4,  # High home office deductions
        }
        
        # Citation tracking for audit defensibility (enhanced with RAG citations)
        self.citations = []
        
        # RAG confidence thresholds for fallback behavior
        self.rag_confidence_thresholds = {
            "high_confidence": ConfidenceLevel.HIGH,
            "medium_confidence": ConfidenceLevel.MEDIUM,
            "fallback_threshold": ConfidenceLevel.LOW  # Below this, use conservative fallback
        }
        
    def get_agent_id(self) -> str:
        """Get the agent's unique identifier."""
        return "categorizer"
    
    async def execute(self, state: WorkflowState) -> WorkflowState:
        """
        Execute tax categorization with strategic optimization and compliance analysis.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated workflow state with categorization results
        """
        try:
            # Validate input data from data processor
            analysis_state = state.get("analysis", {})
            transactions_data = analysis_state.get("transactions", [])
            
            if not transactions_data:
                raise ValueError("No processed transactions available for categorization")
            
            # Convert transaction data to Transaction objects
            transactions = [Transaction(**tx_data) for tx_data in transactions_data]
            
            self.logger.info(f"Categorizing {len(transactions)} transactions with strategic tax optimization")
            
            # Perform strategic tax categorization
            categorized_transactions = await self._categorize_transactions(transactions)
            
            # Calculate tax optimization opportunities
            optimization_analysis = self._analyze_tax_optimization(categorized_transactions)
            
            # Perform compliance risk assessment
            compliance_analysis = self._assess_compliance_risks(categorized_transactions)
            
            # Generate audit-defensible citations
            citation_report = self._generate_citation_report()
            
            # Calculate tax metrics
            tax_metrics = self._calculate_tax_metrics(categorized_transactions)
            
            # Update analysis state with categorization results
            analysis_state.update({
                "transactions": [tx.model_dump() for tx in categorized_transactions],
                "categories": self._generate_category_summary(categorized_transactions),
                "tax_implications": {
                    "total_deductible": tax_metrics["total_deductible"],
                    "optimization_opportunities": optimization_analysis,
                    "compliance_risks": compliance_analysis,
                    "citations": citation_report,
                    "tax_savings_estimate": tax_metrics["estimated_savings"]
                },
                "status": AgentStatus.COMPLETED
            })
            
            # Update workflow state
            state["analysis"] = analysis_state
            
            # Add categorization results to agent memory
            self.update_memory("categorized_transactions", len(categorized_transactions))
            self.update_memory("total_deductible", float(tax_metrics["total_deductible"]))
            self.update_memory("optimization_opportunities", len(optimization_analysis))
            self.update_memory("compliance_risks", len(compliance_analysis))
            
            # Log success with tax strategist's perspective
            self.logger.info(
                f"Strategic tax categorization completed. "
                f"Processed {len(categorized_transactions)} transactions, "
                f"identified ${tax_metrics['total_deductible']:.2f} in deductions, "
                f"found {len(optimization_analysis)} optimization opportunities, "
                f"flagged {len(compliance_analysis)} compliance risks"
            )
            
            return state
            
        except Exception as e:
            self.logger.error(f"Tax categorization failed: {str(e)}")
            # Update analysis state with error
            if "analysis" not in state:
                state["analysis"] = {}
            state["analysis"]["status"] = AgentStatus.FAILED
            state["analysis"]["compliance_issues"] = [str(e)]
            raise
    
    async def _categorize_transactions(self, transactions: List[Transaction]) -> List[Transaction]:
        """
        Categorize transactions using agentic RAG with IRS rule retrieval and fallback to conservative categorization.
        
        Args:
            transactions: List of processed transactions
            
        Returns:
            List of categorized transactions with RAG-enhanced categorization
        """
        categorized_transactions = []
        
        for transaction in transactions:
            try:
                # Apply agentic RAG categorization
                category_info = await self._determine_category_with_rag(transaction)
                tax_category = category_info["name"]
                deduction_rate = category_info["deduction_rate"]
                irs_reference = category_info["irs_reference"]
                confidence = category_info["confidence"]
                rag_citations = category_info.get("rag_citations", [])
                
                # Create updated transaction with categorization
                categorized_tx = Transaction(
                    id=transaction.id,
                    date=transaction.date,
                    amount=transaction.amount,
                    description=transaction.description,
                    account_id=transaction.account_id,
                    counterparty=transaction.counterparty,
                    category=tax_category,
                    tax_category=tax_category,
                    source_file=transaction.source_file,
                    data_quality_score=transaction.data_quality_score,
                    data_issues=transaction.data_issues
                )
                
                categorized_transactions.append(categorized_tx)
                
                # Add enhanced citation for audit defensibility with RAG sources
                citation_entry = {
                    "transaction_id": transaction.id,
                    "category": tax_category,
                    "irs_reference": irs_reference,
                    "deduction_rate": float(deduction_rate),
                    "confidence": confidence.value,
                    "reasoning": category_info.get("reasoning", f"Categorized as {tax_category} using agentic RAG analysis"),
                    "rag_citations": rag_citations,
                    "fallback_used": category_info.get("fallback_used", False)
                }
                
                # Add error field if present
                if "error" in category_info:
                    citation_entry["error"] = category_info["error"]
                
                self.citations.append(citation_entry)
                
                self.logger.debug(f"Categorized transaction {transaction.id} as {tax_category} with {confidence.value} confidence")
                
            except Exception as e:
                self.logger.error(f"Error categorizing transaction {transaction.id}: {str(e)}")
                
                # Conservative fallback for errors
                fallback_category = self.fallback_categories["uncategorized"]
                categorized_tx = Transaction(
                    id=transaction.id,
                    date=transaction.date,
                    amount=transaction.amount,
                    description=transaction.description,
                    account_id=transaction.account_id,
                    counterparty=transaction.counterparty,
                    category=fallback_category["name"],
                    tax_category=fallback_category["name"],
                    source_file=transaction.source_file,
                    data_quality_score=transaction.data_quality_score,
                    data_issues=transaction.data_issues + [f"Categorization error: {str(e)}"]
                )
                
                categorized_transactions.append(categorized_tx)
                
                # Add error citation
                self.citations.append({
                    "transaction_id": transaction.id,
                    "category": fallback_category["name"],
                    "irs_reference": fallback_category["irs_reference"],
                    "deduction_rate": float(fallback_category["deduction_rate"]),
                    "confidence": ConfidenceLevel.VERY_LOW.value,
                    "reasoning": f"Error during categorization, using conservative fallback: {str(e)}",
                    "rag_citations": [],
                    "fallback_used": True,
                    "error": str(e)
                })
        
        return categorized_transactions
    
    async def _determine_category_with_rag(self, transaction: Transaction) -> Dict[str, Any]:
        """
        Determine the appropriate tax category using agentic RAG with IRS rule retrieval.
        Falls back to conservative categorization when RAG confidence is low.
        
        Args:
            transaction: Transaction to categorize
            
        Returns:
            Category information with deduction rate, IRS reference, confidence, and RAG citations
        """
        try:
            # Prepare context for RAG query
            context = {
                "transaction_amount": float(transaction.amount),
                "transaction_date": transaction.date.isoformat(),
                "business_type": "general",  # Could be enhanced with client-specific info
                "description": transaction.description,
                "counterparty": transaction.counterparty or "unknown"
            }
            
            # Use convenience function for expense deduction guidance
            rag_result = await get_expense_deduction_guidance(
                transaction.description,
                float(transaction.amount),
                context.get("business_type", "general")
            )
            
            # Check RAG confidence level
            if rag_result.overall_confidence.value in [ConfidenceLevel.HIGH.value, ConfidenceLevel.MEDIUM.value]:
                # High/medium confidence: use RAG result
                category_info = await self._extract_category_from_rag_result(rag_result, transaction)
                category_info["confidence"] = rag_result.overall_confidence
                category_info["fallback_used"] = False
                category_info["rag_citations"] = self._format_rag_citations(rag_result)
                category_info["reasoning"] = rag_result.synthesized_guidance
                
                self.logger.info(f"RAG categorization successful for transaction {transaction.id} with {rag_result.overall_confidence.value} confidence")
                return category_info
                
            else:
                # Low confidence: fall back to conservative categorization
                self.logger.warning(f"RAG confidence too low ({rag_result.overall_confidence.value}) for transaction {transaction.id}, using fallback")
                return await self._determine_category_fallback(transaction, rag_result)
                
        except Exception as e:
            self.logger.error(f"RAG categorization failed for transaction {transaction.id}: {str(e)}")
            # Fall back to conservative categorization on error
            return await self._determine_category_fallback(transaction, None, error=str(e))
    
    async def _extract_category_from_rag_result(self, rag_result, transaction: Transaction) -> Dict[str, Any]:
        """
        Extract category information from RAG reasoning result.
        
        Args:
            rag_result: Result from agentic RAG reasoning
            transaction: Original transaction
            
        Returns:
            Category information dictionary
        """
        guidance = rag_result.synthesized_guidance.lower()
        
        # Determine category based on RAG guidance
        if "meal" in guidance or "entertainment" in guidance:
            if "50%" in guidance:
                return {
                    "name": "Meals and Entertainment",
                    "deduction_rate": Decimal("0.5"),
                    "irs_reference": self._extract_irs_reference(rag_result)
                }
            elif "not deductible" in guidance:
                return {
                    "name": "Non-Deductible Entertainment",
                    "deduction_rate": Decimal("0.0"),
                    "irs_reference": self._extract_irs_reference(rag_result)
                }
        
        elif "travel" in guidance:
            return {
                "name": "Travel Expenses",
                "deduction_rate": Decimal("1.0"),
                "irs_reference": self._extract_irs_reference(rag_result)
            }
        
        elif "office" in guidance or "supplies" in guidance:
            return {
                "name": "Office Expenses",
                "deduction_rate": Decimal("1.0"),
                "irs_reference": self._extract_irs_reference(rag_result)
            }
        
        elif "equipment" in guidance or "depreciation" in guidance:
            return {
                "name": "Equipment and Depreciation",
                "deduction_rate": Decimal("1.0"),
                "irs_reference": self._extract_irs_reference(rag_result)
            }
        
        elif "professional" in guidance or "legal" in guidance or "accounting" in guidance:
            return {
                "name": "Professional Services",
                "deduction_rate": Decimal("1.0"),
                "irs_reference": self._extract_irs_reference(rag_result)
            }
        
        elif "utility" in guidance or "utilities" in guidance:
            return {
                "name": "Utilities",
                "deduction_rate": Decimal("1.0"),
                "irs_reference": self._extract_irs_reference(rag_result)
            }
        
        elif "marketing" in guidance or "advertising" in guidance:
            return {
                "name": "Marketing and Advertising",
                "deduction_rate": Decimal("1.0"),
                "irs_reference": self._extract_irs_reference(rag_result)
            }
        
        elif "insurance" in guidance:
            return {
                "name": "Insurance",
                "deduction_rate": Decimal("1.0"),
                "irs_reference": self._extract_irs_reference(rag_result)
            }
        
        elif "rent" in guidance or "lease" in guidance:
            return {
                "name": "Rent and Lease",
                "deduction_rate": Decimal("1.0"),
                "irs_reference": self._extract_irs_reference(rag_result)
            }
        
        elif "deductible" in guidance:
            # General business expense
            return {
                "name": "General Business Expenses",
                "deduction_rate": Decimal("1.0"),
                "irs_reference": self._extract_irs_reference(rag_result)
            }
        
        else:
            # Conservative approach when unclear
            return {
                "name": "Uncategorized - Requires Review",
                "deduction_rate": Decimal("0.0"),
                "irs_reference": self._extract_irs_reference(rag_result) or "Manual review required"
            }
    
    def _extract_irs_reference(self, rag_result) -> str:
        """Extract IRS reference from RAG result citations."""
        if not rag_result.rule_interpretations:
            return "RAG analysis - no specific citation"
        
        # Get the highest confidence interpretation
        best_interp = max(rag_result.rule_interpretations, 
                         key=lambda x: x.citations[0].confidence_score if x.citations else 0)
        
        if best_interp.citations:
            citation = best_interp.citations[0]
            ref_parts = []
            if citation.document_title:
                ref_parts.append(citation.document_title)
            if citation.section:
                ref_parts.append(f"Section {citation.section}")
            return ", ".join(ref_parts) if ref_parts else citation.document_id
        
        return "RAG analysis - citation unavailable"
    
    def _format_rag_citations(self, rag_result) -> List[Dict[str, Any]]:
        """Format RAG citations for audit trail."""
        formatted_citations = []
        
        for interp in rag_result.rule_interpretations:
            for citation in interp.citations:
                formatted_citations.append({
                    "document_id": citation.document_id,
                    "document_title": citation.document_title,
                    "section": citation.section,
                    "confidence_score": citation.confidence_score,
                    "publication_year": citation.publication_year,
                    "interpretation": interp.interpretation
                })
        
        return formatted_citations
    
    async def _determine_category_fallback(self, transaction: Transaction, rag_result=None, error=None) -> Dict[str, Any]:
        """
        Conservative fallback categorization when RAG confidence is low or fails.
        
        Args:
            transaction: Transaction to categorize
            rag_result: Optional RAG result for additional context
            error: Optional error message
            
        Returns:
            Conservative category information
        """
        # Use original rule-based approach as fallback
        fallback_category = self._determine_category_fallback_rules(transaction)
        
        # Add fallback metadata
        fallback_category["confidence"] = ConfidenceLevel.LOW
        fallback_category["fallback_used"] = True
        fallback_category["rag_citations"] = []
        
        if rag_result:
            fallback_category["reasoning"] = f"RAG confidence too low ({rag_result.overall_confidence.value}), using conservative fallback: {fallback_category['name']}"
            # Include RAG citations even for fallback for audit trail
            fallback_category["rag_citations"] = self._format_rag_citations(rag_result)
        elif error:
            fallback_category["reasoning"] = f"RAG error ({error}), using conservative fallback: {fallback_category['name']}"
            fallback_category["error"] = error
        else:
            fallback_category["reasoning"] = f"Using conservative fallback categorization: {fallback_category['name']}"
        
        return fallback_category
    
    def _determine_category_fallback_rules(self, transaction: Transaction) -> Dict[str, Any]:
        """
        Original rule-based categorization logic for fallback scenarios.
        
        Args:
            transaction: Transaction to categorize
            
        Returns:
            Category information with deduction rate and IRS reference
        """
        description_lower = transaction.description.lower()
        counterparty_lower = (transaction.counterparty or "").lower()
        combined_text = f"{description_lower} {counterparty_lower}".strip()
        
        # Score each category based on keyword matches with priority weighting
        category_scores = {}
        
        for category_key, category_info in self.fallback_categories.items():
            if category_key == "uncategorized":
                continue
                
            score = 0
            keywords = category_info["keywords"]
            priority_keywords = category_info.get("priority_keywords", [])
            
            # Priority keywords get higher weight
            for keyword in priority_keywords:
                if keyword in combined_text:
                    score += 3  # Higher weight for priority keywords
                    
            # Regular keywords get standard weight
            for keyword in keywords:
                if keyword not in priority_keywords and keyword in combined_text:
                    score += 1
                    
            if score > 0:
                category_scores[category_key] = score
        
        # Select the category with the highest score
        if category_scores:
            best_category = max(category_scores.keys(), key=lambda k: category_scores[k])
            return self.fallback_categories[best_category].copy()
        else:
            # Conservative approach: uncategorized for manual review
            return self.fallback_categories["uncategorized"].copy()
    
    def _analyze_tax_optimization(self, transactions: List[Transaction]) -> List[Dict[str, Any]]:
        """
        Analyze transactions for tax optimization opportunities using RAG-enhanced categorization insights.
        
        Args:
            transactions: Categorized transactions
            
        Returns:
            List of optimization opportunities with RAG-derived insights
        """
        opportunities = []
        
        # Group transactions by category
        category_totals = {}
        rag_insights = {}  # Track RAG insights by category
        
        for tx in transactions:
            category = tx.tax_category or "Uncategorized"
            if category not in category_totals:
                category_totals[category] = Decimal("0")
                rag_insights[category] = []
            category_totals[category] += tx.amount
            
            # Collect RAG insights from citations
            tx_citation = next((c for c in self.citations if c["transaction_id"] == tx.id), None)
            if tx_citation and tx_citation.get("rag_citations"):
                rag_insights[category].extend(tx_citation["rag_citations"])
        
        # Analyze meal expenses for optimization with RAG insights
        meal_total = category_totals.get("Meals and Entertainment", Decimal("0"))
        if meal_total > Decimal("50"):  # Lower threshold for realistic testing
            meal_insights = rag_insights.get("Meals and Entertainment", [])
            irs_ref = "IRS Publication 334, Chapter 11"  # Default
            
            # Use RAG citation if available
            if meal_insights:
                best_citation = max(meal_insights, key=lambda x: x.get("confidence_score", 0))
                if best_citation.get("document_title"):
                    irs_ref = best_citation["document_title"]
                    if best_citation.get("section"):
                        irs_ref += f", {best_citation['section']}"
            
            opportunities.append({
                "type": "meal_deduction_optimization",
                "description": f"Meal expenses of ${meal_total:.2f} qualify for 50% business deduction",
                "potential_savings": float(meal_total * Decimal("0.5") * Decimal("0.25")),  # Assume 25% tax rate
                "action": "Ensure proper documentation for business purpose of meals",
                "irs_reference": irs_ref,
                "rag_enhanced": bool(meal_insights)
            })
        
        # Analyze equipment purchases for depreciation vs expensing
        equipment_total = category_totals.get("Equipment and Depreciation", Decimal("0"))
        if equipment_total > Decimal("2500"):
            equipment_insights = rag_insights.get("Equipment and Depreciation", [])
            irs_ref = "IRS Publication 334, Chapter 9"  # Default
            
            # Use RAG citation if available
            if equipment_insights:
                best_citation = max(equipment_insights, key=lambda x: x.get("confidence_score", 0))
                if best_citation.get("document_title"):
                    irs_ref = best_citation["document_title"]
                    if best_citation.get("section"):
                        irs_ref += f", {best_citation['section']}"
            
            opportunities.append({
                "type": "section_179_deduction",
                "description": f"Equipment purchases of ${equipment_total:.2f} may qualify for Section 179 immediate expensing",
                "potential_savings": float(equipment_total * Decimal("0.25")),  # Assume 25% tax rate
                "action": "Consider Section 179 election for immediate deduction vs depreciation",
                "irs_reference": irs_ref,
                "rag_enhanced": bool(equipment_insights)
            })
        
        # Analyze large transactions for potential splitting
        large_transactions = [tx for tx in transactions if tx.amount >= self.optimization_rules["large_transaction_threshold"]]
        if large_transactions:
            opportunities.append({
                "type": "large_transaction_review",
                "description": f"Found {len(large_transactions)} transactions ≥ $10,000 requiring Form 8300 reporting",
                "potential_savings": 0,
                "action": "Review large transactions for proper reporting requirements and potential audit triggers",
                "irs_reference": "Form 8300 reporting requirements",
                "rag_enhanced": False
            })
        
        # Add RAG-specific optimization opportunities
        rag_opportunities = self._analyze_rag_optimization_insights()
        opportunities.extend(rag_opportunities)
        
        return opportunities
    
    def _analyze_rag_optimization_insights(self) -> List[Dict[str, Any]]:
        """
        Analyze RAG citations for additional optimization opportunities.
        
        Returns:
            List of RAG-derived optimization opportunities
        """
        opportunities = []
        
        # Analyze citations for patterns and insights
        high_confidence_citations = [
            c for c in self.citations 
            if c.get("confidence") in [ConfidenceLevel.HIGH.value, ConfidenceLevel.MEDIUM.value]
            and c.get("rag_citations")
        ]
        
        # Look for conservative categorizations that could be optimized
        conservative_citations = [
            c for c in self.citations
            if c.get("fallback_used") and c.get("rag_citations")
        ]
        
        if conservative_citations:
            opportunities.append({
                "type": "rag_review_opportunity",
                "description": f"Found {len(conservative_citations)} transactions with conservative fallback categorization",
                "potential_savings": 0,  # Would need detailed analysis
                "action": "Review RAG analysis for these transactions - may qualify for higher deductions with additional documentation",
                "irs_reference": "RAG analysis suggests potential optimization",
                "rag_enhanced": True,
                "transaction_count": len(conservative_citations)
            })
        
        # Look for conflicting categorizations that need professional review
        conflicted_citations = [
            c for c in self.citations
            if c.get("confidence") == ConfidenceLevel.LOW.value and c.get("rag_citations")
        ]
        
        if conflicted_citations:
            opportunities.append({
                "type": "professional_review_recommended",
                "description": f"Found {len(conflicted_citations)} transactions with uncertain tax treatment",
                "potential_savings": 0,
                "action": "Professional tax consultation recommended for optimal categorization",
                "irs_reference": "RAG analysis indicates complexity requiring professional guidance",
                "rag_enhanced": True,
                "transaction_count": len(conflicted_citations)
            })
        
        return opportunities
    
    def _assess_compliance_risks(self, transactions: List[Transaction]) -> List[Dict[str, Any]]:
        """
        Assess compliance risks and potential audit triggers using RAG-enhanced analysis.
        
        Args:
            transactions: Categorized transactions
            
        Returns:
            List of compliance risks with RAG insights
        """
        risks = []
        
        # Check for cash transactions requiring Form 8300
        cash_transactions = [tx for tx in transactions if tx.amount >= self.audit_triggers["cash_transaction_limit"]]
        if cash_transactions:
            risks.append({
                "type": "form_8300_requirement",
                "severity": "high",
                "description": f"Found {len(cash_transactions)} transactions ≥ $10,000 requiring Form 8300 reporting",
                "transactions": [tx.id for tx in cash_transactions],
                "action": "File Form 8300 within 15 days of transaction",
                "irs_reference": "Form 8300 Instructions",
                "rag_enhanced": False
            })
        
        # Check for high percentage of round numbers (potential audit trigger)
        round_amounts = sum(1 for tx in transactions if float(tx.amount) % 1 == 0)
        round_percentage = round_amounts / len(transactions) if transactions else 0
        
        if round_percentage > self.audit_triggers["round_number_percentage"]:
            risks.append({
                "type": "round_number_pattern",
                "severity": "medium",
                "description": f"High percentage of round amounts ({round_percentage:.1%}) may trigger audit attention",
                "action": "Ensure proper documentation for round-number transactions",
                "irs_reference": "General audit risk factors",
                "rag_enhanced": False
            })
        
        # Check for high entertainment expenses with RAG insights
        entertainment_total = sum(tx.amount for tx in transactions if tx.tax_category == "Meals and Entertainment")
        total_expenses = sum(tx.amount for tx in transactions)
        entertainment_percentage = float(entertainment_total / total_expenses) if total_expenses > 0 else 0
        
        if entertainment_percentage > self.audit_triggers["entertainment_percentage"]:
            # Get RAG insights for entertainment expenses
            entertainment_citations = [
                c for c in self.citations 
                if "meal" in c.get("category", "").lower() or "entertainment" in c.get("category", "").lower()
            ]
            
            irs_ref = "IRS Publication 334, Chapter 11"  # Default
            rag_enhanced = False
            
            if entertainment_citations:
                rag_enhanced = True
                # Use best RAG citation
                best_citation = max(entertainment_citations, 
                                  key=lambda x: max([rc.get("confidence_score", 0) for rc in x.get("rag_citations", [])], default=0))
                if best_citation.get("rag_citations"):
                    best_rag = max(best_citation["rag_citations"], key=lambda x: x.get("confidence_score", 0))
                    if best_rag.get("document_title"):
                        irs_ref = best_rag["document_title"]
                        if best_rag.get("section"):
                            irs_ref += f", {best_rag['section']}"
            
            risks.append({
                "type": "high_entertainment_expenses",
                "severity": "medium", 
                "description": f"Entertainment expenses represent {entertainment_percentage:.1%} of total expenses",
                "action": "Ensure detailed documentation of business purpose for all entertainment expenses",
                "irs_reference": irs_ref,
                "rag_enhanced": rag_enhanced
            })
        
        # Check for uncategorized transactions
        uncategorized = [tx for tx in transactions if tx.tax_category == "Uncategorized - Requires Review"]
        if uncategorized:
            risks.append({
                "type": "uncategorized_transactions",
                "severity": "low",
                "description": f"Found {len(uncategorized)} uncategorized transactions requiring manual review",
                "transactions": [tx.id for tx in uncategorized],
                "action": "Review and properly categorize all transactions for maximum deduction",
                "irs_reference": "Conservative compliance approach",
                "rag_enhanced": False
            })
        
        # RAG-specific compliance risks
        rag_risks = self._assess_rag_compliance_risks(transactions)
        risks.extend(rag_risks)
        
        return risks
    
    def _assess_rag_compliance_risks(self, transactions: List[Transaction]) -> List[Dict[str, Any]]:
        """
        Assess compliance risks specific to RAG analysis results.
        
        Args:
            transactions: Categorized transactions
            
        Returns:
            List of RAG-specific compliance risks
        """
        risks = []
        
        # Check for high percentage of fallback categorizations
        fallback_citations = [c for c in self.citations if c.get("fallback_used")]
        fallback_percentage = len(fallback_citations) / len(self.citations) if self.citations else 0
        
        if fallback_percentage > 0.3:  # More than 30% fallback
            risks.append({
                "type": "high_fallback_categorization",
                "severity": "medium",
                "description": f"High percentage ({fallback_percentage:.1%}) of transactions used conservative fallback categorization",
                "action": "Review RAG system performance and consider updating IRS document database",
                "irs_reference": "RAG system analysis",
                "rag_enhanced": True,
                "transaction_count": len(fallback_citations)
            })
        
        # Check for low confidence categorizations
        low_confidence_citations = [
            c for c in self.citations 
            if c.get("confidence") in [ConfidenceLevel.LOW.value, ConfidenceLevel.VERY_LOW.value]
        ]
        
        if len(low_confidence_citations) > len(self.citations) * 0.2:  # More than 20% low confidence
            risks.append({
                "type": "low_confidence_categorizations",
                "severity": "medium",
                "description": f"Found {len(low_confidence_citations)} transactions with low confidence categorization",
                "action": "Professional review recommended for low-confidence categorizations",
                "irs_reference": "RAG confidence analysis",
                "rag_enhanced": True,
                "transaction_count": len(low_confidence_citations)
            })
        
        # Check for RAG system errors
        error_citations = [c for c in self.citations if c.get("error")]
        if error_citations:
            risks.append({
                "type": "rag_system_errors",
                "severity": "high",
                "description": f"RAG system encountered errors processing {len(error_citations)} transactions",
                "action": "Review RAG system configuration and IRS document database integrity",
                "irs_reference": "System error analysis",
                "rag_enhanced": True,
                "transaction_count": len(error_citations),
                "errors": [c.get("error") for c in error_citations]
            })
        
        return risks
    
    def _generate_citation_report(self) -> List[Dict[str, Any]]:
        """
        Generate comprehensive audit-defensible citation report with RAG sources.
        
        Returns:
            List of citations with IRS references and RAG analysis details
        """
        enhanced_citations = []
        
        for citation in self.citations:
            enhanced_citation = citation.copy()
            
            # Add RAG analysis summary
            if citation.get("rag_citations"):
                rag_summary = {
                    "rag_sources_count": len(citation["rag_citations"]),
                    "highest_confidence_source": max(
                        citation["rag_citations"], 
                        key=lambda x: x.get("confidence_score", 0)
                    ) if citation["rag_citations"] else None,
                    "document_sources": list(set(
                        rc.get("document_title", "Unknown") 
                        for rc in citation["rag_citations"]
                    ))
                }
                enhanced_citation["rag_analysis"] = rag_summary
            
            # Add compliance flags
            enhanced_citation["compliance_flags"] = {
                "high_confidence": citation.get("confidence") == ConfidenceLevel.HIGH.value,
                "rag_enhanced": bool(citation.get("rag_citations")),
                "fallback_used": citation.get("fallback_used", False),
                "professional_review_recommended": citation.get("confidence") in [
                    ConfidenceLevel.LOW.value, ConfidenceLevel.VERY_LOW.value
                ]
            }
            
            enhanced_citations.append(enhanced_citation)
        
        return enhanced_citations
    
    def _calculate_tax_metrics(self, transactions: List[Transaction]) -> Dict[str, Decimal]:
        """
        Calculate comprehensive tax metrics and savings estimates.
        
        Args:
            transactions: Categorized transactions
            
        Returns:
            Dictionary of tax metrics
        """
        total_expenses = sum(tx.amount for tx in transactions)
        total_deductible = Decimal("0")
        
        # Calculate deductible amounts by category
        for tx in transactions:
            category_key = self._get_category_key(tx.tax_category)
            if category_key in self.fallback_categories:
                deduction_rate = self.fallback_categories[category_key]["deduction_rate"]
                total_deductible += tx.amount * deduction_rate
        
        # Estimate tax savings (assume 25% effective tax rate)
        estimated_savings = total_deductible * Decimal("0.25")
        
        return {
            "total_expenses": total_expenses,
            "total_deductible": total_deductible,
            "estimated_savings": estimated_savings,
            "deduction_percentage": float(total_deductible / total_expenses) if total_expenses > 0 else 0
        }
    
    def _get_category_key(self, category_name: str) -> str:
        """
        Get the category key from category name.
        
        Args:
            category_name: Display name of category
            
        Returns:
            Category key for lookup
        """
        for key, info in self.fallback_categories.items():
            if info["name"] == category_name:
                return key
        return "uncategorized"
    
    def _generate_category_summary(self, transactions: List[Transaction]) -> Dict[str, str]:
        """
        Generate summary of categories assigned to transactions.
        
        Args:
            transactions: Categorized transactions
            
        Returns:
            Dictionary mapping transaction IDs to categories
        """
        return {tx.id: tx.tax_category or "Uncategorized" for tx in transactions}