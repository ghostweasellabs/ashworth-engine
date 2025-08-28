"""
Categorizer Agent - Clarke Pemberton, JD, CPA
Corporate Tax Compliance Strategist with strategic tax optimization mindset.
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


class CategorizerAgent(BaseAgent):
    """
    Clarke Pemberton, JD, CPA - Corporate Tax Compliance Strategist
    
    Persona: Big Four accounting firm veteran with JD and CPA credentials,
    specializing in strategic tax optimization and IRS compliance with
    meticulous attention to audit defensibility.
    """
    
    def __init__(self):
        super().__init__(CATEGORIZER_PERSONALITY)
        
        # Basic IRS expense categories (will be enhanced with RAG later)
        self.irs_categories = {
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
        
        # Citation tracking for audit defensibility
        self.citations = []
        
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
        Categorize transactions using rule-based approach with strategic tax optimization.
        
        Args:
            transactions: List of processed transactions
            
        Returns:
            List of categorized transactions
        """
        categorized_transactions = []
        
        for transaction in transactions:
            # Apply strategic categorization logic
            category_info = self._determine_category(transaction)
            tax_category = category_info["name"]
            deduction_rate = category_info["deduction_rate"]
            irs_reference = category_info["irs_reference"]
            
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
            
            # Add citation for audit defensibility
            self.citations.append({
                "transaction_id": transaction.id,
                "category": tax_category,
                "irs_reference": irs_reference,
                "deduction_rate": float(deduction_rate),
                "reasoning": f"Categorized as {tax_category} based on description analysis and IRS guidelines"
            })
        
        return categorized_transactions
    
    def _determine_category(self, transaction: Transaction) -> Dict[str, Any]:
        """
        Determine the appropriate tax category for a transaction using conservative compliance approach.
        
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
        
        for category_key, category_info in self.irs_categories.items():
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
            return self.irs_categories[best_category]
        else:
            # Conservative approach: uncategorized for manual review
            return self.irs_categories["uncategorized"]
    
    def _analyze_tax_optimization(self, transactions: List[Transaction]) -> List[Dict[str, Any]]:
        """
        Analyze transactions for tax optimization opportunities.
        
        Args:
            transactions: Categorized transactions
            
        Returns:
            List of optimization opportunities
        """
        opportunities = []
        
        # Group transactions by category
        category_totals = {}
        for tx in transactions:
            category = tx.tax_category or "Uncategorized"
            if category not in category_totals:
                category_totals[category] = Decimal("0")
            category_totals[category] += tx.amount
        
        # Analyze meal expenses for optimization
        meal_total = category_totals.get("Meals and Entertainment", Decimal("0"))
        if meal_total > Decimal("50"):  # Lower threshold for realistic testing
            opportunities.append({
                "type": "meal_deduction_optimization",
                "description": f"Meal expenses of ${meal_total:.2f} qualify for 50% business deduction",
                "potential_savings": float(meal_total * Decimal("0.5") * Decimal("0.25")),  # Assume 25% tax rate
                "action": "Ensure proper documentation for business purpose of meals",
                "irs_reference": "IRS Publication 334, Chapter 11"
            })
        
        # Analyze equipment purchases for depreciation vs expensing
        equipment_total = category_totals.get("Equipment and Depreciation", Decimal("0"))
        if equipment_total > Decimal("2500"):
            opportunities.append({
                "type": "section_179_deduction",
                "description": f"Equipment purchases of ${equipment_total:.2f} may qualify for Section 179 immediate expensing",
                "potential_savings": float(equipment_total * Decimal("0.25")),  # Assume 25% tax rate
                "action": "Consider Section 179 election for immediate deduction vs depreciation",
                "irs_reference": "IRS Publication 334, Chapter 9"
            })
        
        # Analyze large transactions for potential splitting
        large_transactions = [tx for tx in transactions if tx.amount >= self.optimization_rules["large_transaction_threshold"]]
        if large_transactions:
            opportunities.append({
                "type": "large_transaction_review",
                "description": f"Found {len(large_transactions)} transactions ≥ $10,000 requiring Form 8300 reporting",
                "potential_savings": 0,
                "action": "Review large transactions for proper reporting requirements and potential audit triggers",
                "irs_reference": "Form 8300 reporting requirements"
            })
        
        return opportunities
    
    def _assess_compliance_risks(self, transactions: List[Transaction]) -> List[Dict[str, Any]]:
        """
        Assess compliance risks and potential audit triggers.
        
        Args:
            transactions: Categorized transactions
            
        Returns:
            List of compliance risks
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
                "irs_reference": "Form 8300 Instructions"
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
                "irs_reference": "General audit risk factors"
            })
        
        # Check for high entertainment expenses
        entertainment_total = sum(tx.amount for tx in transactions if tx.tax_category == "Meals and Entertainment")
        total_expenses = sum(tx.amount for tx in transactions)
        entertainment_percentage = float(entertainment_total / total_expenses) if total_expenses > 0 else 0
        
        if entertainment_percentage > self.audit_triggers["entertainment_percentage"]:
            risks.append({
                "type": "high_entertainment_expenses",
                "severity": "medium", 
                "description": f"Entertainment expenses represent {entertainment_percentage:.1%} of total expenses",
                "action": "Ensure detailed documentation of business purpose for all entertainment expenses",
                "irs_reference": "IRS Publication 334, Chapter 11"
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
                "irs_reference": "Conservative compliance approach"
            })
        
        return risks
    
    def _generate_citation_report(self) -> List[Dict[str, Any]]:
        """
        Generate audit-defensible citation report for all categorization decisions.
        
        Returns:
            List of citations with IRS references
        """
        return self.citations.copy()
    
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
            if category_key in self.irs_categories:
                deduction_rate = self.irs_categories[category_key]["deduction_rate"]
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
        for key, info in self.irs_categories.items():
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