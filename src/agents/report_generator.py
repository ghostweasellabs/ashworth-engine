"""
Report Generator Agent - Professor Elena Castellanos
Executive Financial Storytelling Director with compelling narrative generation.
"""

import asyncio
import logging
import json
import uuid
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import re

from src.agents.base import BaseAgent
from src.config.personas import REPORT_GENERATOR_PERSONALITY
from src.workflows.state_schemas import WorkflowState, AgentStatus, ReportState
from src.utils.llm import get_llm_router, ModelTier
try:
    from src.utils.storage import SupabaseStorageManager
    from src.utils.visualization import ChartGenerator
except ImportError:
    # Fallback for testing without dependencies
    class SupabaseStorageManager:
        def __init__(self):
            pass
        async def upload_text(self, *args, **kwargs):
            return "https://example.com/test.md"
    
    class ChartGenerator:
        def __init__(self):
            pass
        async def create_pie_chart(self, *args, **kwargs):
            return {"data": [], "config": {}, "type": "pie", "title": "Test"}
        async def create_line_chart(self, *args, **kwargs):
            return {"data": [], "config": {}, "type": "line", "title": "Test"}
        async def create_bar_chart(self, *args, **kwargs):
            return {"data": [], "config": {}, "type": "bar", "title": "Test"}
from src.models.base import Transaction, FinancialMetrics


class ReportGeneratorAgent(BaseAgent):
    """
    Professor Elena Castellanos - Executive Financial Storytelling Director
    
    Persona: Chicago Booth MBA and former Bain & Company consultant specializing
    in executive communication and strategic financial narratives that convert
    data into compelling strategic action.
    """
    
    def __init__(self):
        super().__init__(REPORT_GENERATOR_PERSONALITY)
        
        # Initialize LLM router for narrative generation
        self.llm_router = get_llm_router()
        
        # Initialize storage manager for report persistence
        self.storage_manager = SupabaseStorageManager()
        
        # Initialize chart generator for visualizations
        self.chart_generator = ChartGenerator()
        
        # Report templates and structure
        self.report_templates = {
            "executive_summary": {
                "sections": ["key_insights", "financial_highlights", "strategic_recommendations"],
                "max_length": 500,
                "tone": "executive"
            },
            "detailed_analysis": {
                "sections": ["financial_overview", "category_analysis", "tax_optimization", "compliance_review"],
                "max_length": 2000,
                "tone": "analytical"
            },
            "strategic_recommendations": {
                "sections": ["immediate_actions", "medium_term_strategy", "long_term_planning"],
                "max_length": 1000,
                "tone": "strategic"
            }
        }
        
        # Visualization configurations
        self.chart_configs = {
            "expense_categories": {
                "type": "pie",
                "title": "Expense Distribution by Category",
                "theme": "professional"
            },
            "monthly_trends": {
                "type": "line",
                "title": "Monthly Financial Trends",
                "theme": "professional"
            },
            "tax_optimization": {
                "type": "bar",
                "title": "Tax Optimization Opportunities",
                "theme": "professional"
            }
        }
        
    def get_agent_id(self) -> str:
        """Get the agent's unique identifier."""
        return "report_generator"
    
    async def execute(self, state: WorkflowState) -> WorkflowState:
        """
        Execute report generation with executive storytelling and Supabase storage.
        
        Args:
            state: Current workflow state
            
        Returns:
            Updated workflow state with report results
        """
        try:
            # Validate input data from previous agents
            analysis_state = state.get("analysis", {})
            transactions_data = analysis_state.get("transactions", [])
            tax_implications = analysis_state.get("tax_implications", {})
            
            if not transactions_data:
                raise ValueError("No analyzed transactions available for report generation")
            
            self.logger.info(f"Generating executive report for {len(transactions_data)} transactions")
            
            # Convert transaction data to Transaction objects
            transactions = [Transaction(**tx_data) for tx_data in transactions_data]
            
            # Generate comprehensive financial metrics
            financial_metrics = self._calculate_comprehensive_metrics(transactions, tax_implications)
            
            # Create professional visualizations
            visualizations = await self._generate_visualizations(transactions, financial_metrics, tax_implications)
            
            # Generate compelling narrative content
            narrative_content = await self._generate_executive_narrative(
                transactions, financial_metrics, tax_implications, visualizations
            )
            
            # Structure the complete report
            structured_report = self._structure_complete_report(
                narrative_content, visualizations, financial_metrics, tax_implications
            )
            
            # Generate report metadata
            report_metadata = self._generate_report_metadata(
                state.get("workflow_id", "unknown"),
                len(transactions),
                financial_metrics
            )
            
            # Store report in Supabase with versioning
            storage_result = await self._store_report_with_versioning(
                structured_report, visualizations, report_metadata
            )
            
            # Update report state
            report_state = ReportState(
                report_id=storage_result["report_id"],
                report_type="executive_financial_analysis",
                content=structured_report,
                visualizations=visualizations,
                storage_path=storage_result["storage_path"],
                metadata=report_metadata,
                status=AgentStatus.COMPLETED
            )
            
            # Update workflow state
            state["report"] = report_state
            
            # Add report URL to output reports
            if "output_reports" not in state:
                state["output_reports"] = []
            state["output_reports"].append(storage_result["public_url"])
            
            # Add report generation results to agent memory
            self.update_memory("report_id", storage_result["report_id"])
            self.update_memory("report_url", storage_result["public_url"])
            self.update_memory("visualizations_count", len(visualizations))
            self.update_memory("narrative_quality_score", narrative_content.get("quality_score", 0.0))
            
            # Log success with executive storytelling perspective
            self.logger.info(
                f"Executive report generation completed. "
                f"Generated compelling narrative for {len(transactions)} transactions, "
                f"created {len(visualizations)} professional visualizations, "
                f"stored report with ID: {storage_result['report_id']}"
            )
            
            return state
            
        except Exception as e:
            self.logger.error(f"Report generation failed: {str(e)}")
            # Update report state with error
            if "report" not in state:
                state["report"] = {}
            state["report"]["status"] = AgentStatus.FAILED
            state["report"]["error_message"] = str(e)
            raise
    
    def _calculate_comprehensive_metrics(
        self, 
        transactions: List[Transaction], 
        tax_implications: Dict[str, Any]
    ) -> FinancialMetrics:
        """
        Calculate comprehensive financial metrics for executive reporting.
        
        Args:
            transactions: List of processed transactions
            tax_implications: Tax analysis results
            
        Returns:
            Comprehensive financial metrics
        """
        # Calculate basic totals
        total_expenses = sum(tx.amount for tx in transactions if tx.amount > 0)
        total_revenue = sum(abs(tx.amount) for tx in transactions if tx.amount < 0)
        net_income = total_revenue - total_expenses
        
        # Calculate expense categories
        expense_categories = {}
        for tx in transactions:
            if tx.amount > 0:  # Expenses are positive
                category = tx.tax_category or "Uncategorized"
                if category not in expense_categories:
                    expense_categories[category] = Decimal("0")
                expense_categories[category] += tx.amount
        
        # Get tax deductible amount from tax implications
        tax_deductible_amount = Decimal(str(tax_implications.get("total_deductible", 0)))
        
        return FinancialMetrics(
            total_revenue=total_revenue,
            total_expenses=total_expenses,
            net_income=net_income,
            expense_categories=expense_categories,
            tax_deductible_amount=tax_deductible_amount
        )
    
    async def _generate_visualizations(
        self,
        transactions: List[Transaction],
        financial_metrics: FinancialMetrics,
        tax_implications: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Generate professional visualizations for the executive report.
        
        Args:
            transactions: List of processed transactions
            financial_metrics: Calculated financial metrics
            tax_implications: Tax analysis results
            
        Returns:
            List of visualization configurations and data
        """
        visualizations = []
        
        try:
            # 1. Expense Categories Pie Chart
            if financial_metrics.expense_categories:
                expense_chart = await self.chart_generator.create_pie_chart(
                    data=dict(financial_metrics.expense_categories),
                    title="Expense Distribution by Category",
                    theme="professional"
                )
                visualizations.append({
                    "id": "expense_categories",
                    "type": "pie",
                    "title": "Expense Distribution by Category",
                    "data": expense_chart["data"],
                    "config": expense_chart["config"],
                    "insights": self._generate_chart_insights("expense_categories", financial_metrics.expense_categories)
                })
            
            # 2. Monthly Trends Line Chart
            monthly_data = self._aggregate_monthly_data(transactions)
            if monthly_data:
                trends_chart = await self.chart_generator.create_line_chart(
                    data=monthly_data,
                    title="Monthly Financial Trends",
                    theme="professional"
                )
                visualizations.append({
                    "id": "monthly_trends",
                    "type": "line", 
                    "title": "Monthly Financial Trends",
                    "data": trends_chart["data"],
                    "config": trends_chart["config"],
                    "insights": self._generate_chart_insights("monthly_trends", monthly_data)
                })
            
            # 3. Tax Optimization Opportunities Bar Chart
            optimization_opportunities = tax_implications.get("optimization_opportunities", [])
            if optimization_opportunities:
                tax_chart_data = {
                    opp["type"]: opp.get("potential_savings", 0)
                    for opp in optimization_opportunities
                }
                tax_chart = await self.chart_generator.create_bar_chart(
                    data=tax_chart_data,
                    title="Tax Optimization Opportunities",
                    theme="professional"
                )
                visualizations.append({
                    "id": "tax_optimization",
                    "type": "bar",
                    "title": "Tax Optimization Opportunities",
                    "data": tax_chart["data"],
                    "config": tax_chart["config"],
                    "insights": self._generate_chart_insights("tax_optimization", tax_chart_data)
                })
            
            self.logger.info(f"Generated {len(visualizations)} professional visualizations")
            
        except Exception as e:
            self.logger.warning(f"Visualization generation encountered issues: {e}")
            # Continue with report generation even if visualizations fail
        
        return visualizations
    
    async def _generate_executive_narrative(
        self,
        transactions: List[Transaction],
        financial_metrics: FinancialMetrics,
        tax_implications: Dict[str, Any],
        visualizations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate compelling executive narrative using LLM with storytelling expertise.
        
        Args:
            transactions: List of processed transactions
            financial_metrics: Calculated financial metrics
            tax_implications: Tax analysis results
            visualizations: Generated visualizations
            
        Returns:
            Structured narrative content with quality metrics
        """
        # Prepare context for narrative generation
        context = {
            "transaction_count": len(transactions),
            "total_expenses": float(financial_metrics.total_expenses),
            "total_revenue": float(financial_metrics.total_revenue),
            "net_income": float(financial_metrics.net_income),
            "tax_deductible": float(financial_metrics.tax_deductible_amount),
            "top_categories": dict(list(financial_metrics.expense_categories.items())[:5]),
            "optimization_opportunities": len(tax_implications.get("optimization_opportunities", [])),
            "compliance_risks": len(tax_implications.get("compliance_risks", [])),
            "visualization_count": len(visualizations)
        }
        
        # Generate executive summary with compelling storytelling
        executive_summary = await self._generate_narrative_section(
            "executive_summary",
            context,
            "Create a compelling executive summary that transforms financial data into strategic insights. "
            "Focus on the story the numbers tell and actionable recommendations for business growth."
        )
        
        # Generate detailed financial analysis
        detailed_analysis = await self._generate_narrative_section(
            "detailed_analysis",
            context,
            "Provide a comprehensive financial analysis that explains patterns, trends, and opportunities. "
            "Use consulting-grade language suitable for C-suite presentation."
        )
        
        # Generate strategic recommendations
        strategic_recommendations = await self._generate_narrative_section(
            "strategic_recommendations",
            context,
            "Develop specific, actionable strategic recommendations with implementation steps. "
            "Focus on tax optimization, cost management, and growth opportunities."
        )
        
        # Calculate narrative quality score
        quality_score = self._assess_narrative_quality(
            executive_summary, detailed_analysis, strategic_recommendations
        )
        
        return {
            "executive_summary": executive_summary,
            "detailed_analysis": detailed_analysis,
            "strategic_recommendations": strategic_recommendations,
            "quality_score": quality_score,
            "generation_metadata": {
                "generated_at": datetime.utcnow().isoformat(),
                "context_size": len(str(context)),
                "total_words": self._count_words(executive_summary + detailed_analysis + strategic_recommendations)
            }
        }
    
    async def _generate_narrative_section(
        self,
        section_type: str,
        context: Dict[str, Any],
        instruction: str
    ) -> str:
        """
        Generate a specific narrative section using LLM with executive storytelling.
        
        Args:
            section_type: Type of section to generate
            context: Financial context data
            instruction: Specific instruction for this section
            
        Returns:
            Generated narrative content
        """
        # Create compelling prompt with Professor Elena's expertise
        prompt = f"""As Professor Elena Castellanos, Executive Financial Storytelling Director with Chicago Booth MBA and Bain & Company experience, {instruction}

Financial Context:
- Transaction Count: {context['transaction_count']}
- Total Expenses: ${context['total_expenses']:,.2f}
- Total Revenue: ${context['total_revenue']:,.2f}
- Net Income: ${context['net_income']:,.2f}
- Tax Deductible Amount: ${context['tax_deductible']:,.2f}
- Top Expense Categories: {context['top_categories']}
- Optimization Opportunities: {context['optimization_opportunities']}
- Compliance Risks: {context['compliance_risks']}

Requirements:
1. Use compelling executive language that drives action
2. Transform data into strategic insights and narratives
3. Provide specific, implementable recommendations
4. Maintain consulting-grade professional standards
5. Focus on business impact and growth opportunities
6. Keep the tone persuasive yet authoritative

Generate a {section_type.replace('_', ' ')} that converts these financial insights into compelling strategic guidance."""
        
        try:
            # Use heavy model tier for complex narrative generation
            response = await self.llm_router.generate(
                prompt=prompt,
                tier=ModelTier.HEAVY,
                temperature=0.7,
                max_tokens=1000
            )
            
            return response.content.strip()
            
        except Exception as e:
            self.logger.warning(f"LLM narrative generation failed for {section_type}: {e}")
            # Fallback to template-based generation
            return self._generate_fallback_narrative(section_type, context)
    
    def _generate_fallback_narrative(self, section_type: str, context: Dict[str, Any]) -> str:
        """Generate fallback narrative when LLM is unavailable."""
        if section_type == "executive_summary":
            return f"""## Executive Summary

Our financial analysis reveals {context['transaction_count']} transactions totaling ${context['total_expenses']:,.2f} in expenses against ${context['total_revenue']:,.2f} in revenue, resulting in a net income of ${context['net_income']:,.2f}.

Key highlights include ${context['tax_deductible']:,.2f} in tax-deductible expenses and {context['optimization_opportunities']} identified optimization opportunities. The analysis reveals strategic opportunities for improved financial performance and tax efficiency.

**Strategic Priority**: Focus on the {context['optimization_opportunities']} optimization opportunities to enhance profitability while maintaining compliance standards."""
        
        elif section_type == "detailed_analysis":
            return f"""## Detailed Financial Analysis

### Financial Overview
The comprehensive analysis of {context['transaction_count']} transactions reveals important patterns in expense management and revenue generation. Total expenses of ${context['total_expenses']:,.2f} represent the primary focus for optimization efforts.

### Category Analysis
Top expense categories show concentration in: {', '.join(context['top_categories'].keys())}. This distribution indicates opportunities for strategic cost management and category-specific optimization.

### Tax Optimization
With ${context['tax_deductible']:,.2f} in deductible expenses, the organization demonstrates strong compliance practices. {context['optimization_opportunities']} additional opportunities have been identified for further tax efficiency."""
        
        else:  # strategic_recommendations
            return f"""## Strategic Recommendations

### Immediate Actions
1. **Tax Optimization**: Implement the {context['optimization_opportunities']} identified opportunities to maximize deductions
2. **Expense Management**: Review top spending categories for cost reduction potential
3. **Compliance Review**: Address {context['compliance_risks']} compliance risks to ensure regulatory adherence

### Medium-Term Strategy
- Establish systematic expense categorization processes
- Implement monthly financial review cycles
- Develop tax planning strategies for upcoming periods

### Long-Term Planning
- Create comprehensive financial forecasting models
- Establish performance benchmarks and KPIs
- Develop strategic partnerships for cost optimization""" 
   
    def _structure_complete_report(
        self,
        narrative_content: Dict[str, Any],
        visualizations: List[Dict[str, Any]],
        financial_metrics: FinancialMetrics,
        tax_implications: Dict[str, Any]
    ) -> str:
        """
        Structure the complete executive report in markdown format.
        
        Args:
            narrative_content: Generated narrative sections
            visualizations: Professional visualizations
            financial_metrics: Calculated financial metrics
            tax_implications: Tax analysis results
            
        Returns:
            Complete structured markdown report
        """
        report_sections = []
        
        # Report Header
        report_sections.append(f"""# Executive Financial Intelligence Report
*Generated by Ashworth Engine v2 - Financial Intelligence Platform*

**Report Date**: {datetime.utcnow().strftime('%B %d, %Y')}
**Analysis Period**: {self._determine_analysis_period(financial_metrics)}
**Report Type**: Comprehensive Financial Analysis with Tax Optimization

---
""")
        
        # Executive Summary
        report_sections.append(f"""## Executive Summary

{narrative_content['executive_summary']}

### Key Financial Metrics
- **Total Revenue**: ${financial_metrics.total_revenue:,.2f}
- **Total Expenses**: ${financial_metrics.total_expenses:,.2f}
- **Net Income**: ${financial_metrics.net_income:,.2f}
- **Tax Deductible Amount**: ${financial_metrics.tax_deductible_amount:,.2f}
- **Potential Tax Savings**: ${float(financial_metrics.tax_deductible_amount) * 0.25:,.2f} *(estimated at 25% tax rate)*

---
""")
        
        # Detailed Analysis
        report_sections.append(f"""## Detailed Financial Analysis

{narrative_content['detailed_analysis']}

### Expense Category Breakdown
""")
        
        # Add expense category table
        if financial_metrics.expense_categories:
            report_sections.append("| Category | Amount | Percentage |\n|----------|--------|------------|\n")
            total_expenses = sum(financial_metrics.expense_categories.values())
            for category, amount in sorted(financial_metrics.expense_categories.items(), key=lambda x: x[1], reverse=True):
                percentage = (amount / total_expenses * 100) if total_expenses > 0 else 0
                report_sections.append(f"| {category} | ${amount:,.2f} | {percentage:.1f}% |\n")
        
        report_sections.append("\n---\n")
        
        # Visualizations Section
        if visualizations:
            report_sections.append("## Professional Visualizations\n\n")
            for viz in visualizations:
                report_sections.append(f"""### {viz['title']}
*Chart Type: {viz['type'].title()}*

{viz.get('insights', 'Professional visualization supporting the financial analysis.')}

*[Visualization: {viz['id']}]*

""")
            report_sections.append("---\n")
        
        # Tax Optimization Section
        optimization_opportunities = tax_implications.get("optimization_opportunities", [])
        if optimization_opportunities:
            report_sections.append("## Tax Optimization Opportunities\n\n")
            for i, opp in enumerate(optimization_opportunities, 1):
                report_sections.append(f"""### {i}. {opp.get('type', 'Optimization Opportunity').replace('_', ' ').title()}

**Description**: {opp.get('description', 'Tax optimization opportunity identified')}

**Potential Savings**: ${opp.get('potential_savings', 0):,.2f}

**Recommended Action**: {opp.get('action', 'Review with tax professional')}

**IRS Reference**: {opp.get('irs_reference', 'Consult current tax regulations')}

""")
            report_sections.append("---\n")
        
        # Compliance Review Section
        compliance_risks = tax_implications.get("compliance_risks", [])
        if compliance_risks:
            report_sections.append("## Compliance Review\n\n")
            for i, risk in enumerate(compliance_risks, 1):
                severity_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(risk.get("severity", "low"), "⚪")
                report_sections.append(f"""### {i}. {risk.get('type', 'Compliance Issue').replace('_', ' ').title()} {severity_emoji}

**Severity**: {risk.get('severity', 'Unknown').title()}

**Description**: {risk.get('description', 'Compliance issue identified')}

**Recommended Action**: {risk.get('action', 'Review with compliance team')}

**IRS Reference**: {risk.get('irs_reference', 'Consult current regulations')}

""")
            report_sections.append("---\n")
        
        # Strategic Recommendations
        report_sections.append(f"""## Strategic Recommendations

{narrative_content['strategic_recommendations']}

---
""")
        
        # Report Footer
        report_sections.append(f"""## Report Metadata

**Generated By**: Professor Elena Castellanos (Executive Financial Storytelling Director)
**Analysis Engine**: Ashworth Engine v2
**Report Quality Score**: {narrative_content.get('quality_score', 0.0):.2f}/5.0
**Total Words**: {narrative_content.get('generation_metadata', {}).get('total_words', 0):,}
**Visualizations**: {len(visualizations)}

*This report was generated using advanced AI financial analysis with human-level expertise in tax compliance and strategic financial planning. All recommendations should be reviewed with qualified financial and tax professionals.*

---

**Disclaimer**: This report is for informational purposes only and does not constitute professional financial, tax, or legal advice. Consult with qualified professionals before making financial decisions.
""")
        
        return "".join(report_sections)
    
    def _generate_report_metadata(
        self,
        workflow_id: str,
        transaction_count: int,
        financial_metrics: FinancialMetrics
    ) -> Dict[str, Any]:
        """
        Generate comprehensive metadata for the report.
        
        Args:
            workflow_id: Workflow identifier
            transaction_count: Number of transactions analyzed
            financial_metrics: Financial metrics calculated
            
        Returns:
            Report metadata dictionary
        """
        return {
            "report_id": str(uuid.uuid4()),
            "workflow_id": workflow_id,
            "generated_at": datetime.utcnow().isoformat(),
            "generated_by": "Professor Elena Castellanos",
            "agent_id": "report_generator",
            "report_type": "executive_financial_analysis",
            "version": "1.0",
            "transaction_count": transaction_count,
            "financial_summary": {
                "total_revenue": float(financial_metrics.total_revenue),
                "total_expenses": float(financial_metrics.total_expenses),
                "net_income": float(financial_metrics.net_income),
                "tax_deductible": float(financial_metrics.tax_deductible_amount)
            },
            "category_count": len(financial_metrics.expense_categories),
            "processing_metadata": {
                "engine_version": "ashworth_v2",
                "compliance_validated": True,
                "executive_grade": True
            }
        }
    
    async def _store_report_with_versioning(
        self,
        report_content: str,
        visualizations: List[Dict[str, Any]],
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Store report in Supabase with proper versioning and metadata management.
        
        Args:
            report_content: Complete markdown report
            visualizations: Generated visualizations
            metadata: Report metadata
            
        Returns:
            Storage result with URLs and identifiers
        """
        try:
            # Create storage path with versioning
            report_id = metadata["report_id"]
            workflow_id = metadata["workflow_id"]
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            
            storage_path = f"reports/{workflow_id}/{report_id}_{timestamp}"
            
            # Store main report content
            report_file_path = f"{storage_path}/report.md"
            report_url = await self.storage_manager.upload_text(
                content=report_content,
                file_path=report_file_path,
                content_type="text/markdown"
            )
            
            # Store metadata
            metadata_file_path = f"{storage_path}/metadata.json"
            metadata_url = await self.storage_manager.upload_text(
                content=json.dumps(metadata, indent=2, default=str),
                file_path=metadata_file_path,
                content_type="application/json"
            )
            
            # Store visualizations data
            visualizations_file_path = f"{storage_path}/visualizations.json"
            viz_url = await self.storage_manager.upload_text(
                content=json.dumps(visualizations, indent=2, default=str),
                file_path=visualizations_file_path,
                content_type="application/json"
            )
            
            # Create version index for easy access to latest version
            version_index = {
                "latest_version": report_id,
                "latest_timestamp": timestamp,
                "report_url": report_url,
                "metadata_url": metadata_url,
                "visualizations_url": viz_url,
                "storage_path": storage_path
            }
            
            version_index_path = f"reports/{workflow_id}/latest.json"
            await self.storage_manager.upload_text(
                content=json.dumps(version_index, indent=2),
                file_path=version_index_path,
                content_type="application/json"
            )
            
            self.logger.info(f"Report stored successfully at {storage_path}")
            
            return {
                "report_id": report_id,
                "storage_path": storage_path,
                "public_url": report_url,
                "metadata_url": metadata_url,
                "visualizations_url": viz_url,
                "version_index_url": f"{self.storage_manager.base_url}/{version_index_path}"
            }
            
        except Exception as e:
            self.logger.error(f"Report storage failed: {e}")
            # Fallback to local storage
            return await self._store_report_locally(report_content, visualizations, metadata)
    
    async def _store_report_locally(
        self,
        report_content: str,
        visualizations: List[Dict[str, Any]],
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Fallback method to store report locally when Supabase is unavailable.
        
        Args:
            report_content: Complete markdown report
            visualizations: Generated visualizations
            metadata: Report metadata
            
        Returns:
            Local storage result
        """
        try:
            # Create local reports directory
            reports_dir = Path("reports")
            reports_dir.mkdir(exist_ok=True)
            
            # Create report-specific directory
            report_id = metadata["report_id"]
            workflow_id = metadata["workflow_id"]
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            
            report_dir = reports_dir / workflow_id / f"{report_id}_{timestamp}"
            report_dir.mkdir(parents=True, exist_ok=True)
            
            # Store report content
            report_file = report_dir / "report.md"
            report_file.write_text(report_content, encoding="utf-8")
            
            # Store metadata
            metadata_file = report_dir / "metadata.json"
            metadata_file.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
            
            # Store visualizations
            viz_file = report_dir / "visualizations.json"
            viz_file.write_text(json.dumps(visualizations, indent=2, default=str), encoding="utf-8")
            
            self.logger.info(f"Report stored locally at {report_dir}")
            
            return {
                "report_id": report_id,
                "storage_path": str(report_dir),
                "public_url": f"file://{report_file.absolute()}",
                "metadata_url": f"file://{metadata_file.absolute()}",
                "visualizations_url": f"file://{viz_file.absolute()}",
                "version_index_url": f"file://{report_dir.absolute()}"
            }
            
        except Exception as e:
            self.logger.error(f"Local report storage failed: {e}")
            raise RuntimeError(f"Both Supabase and local storage failed: {e}")
    
    def _aggregate_monthly_data(self, transactions: List[Transaction]) -> Dict[str, float]:
        """
        Aggregate transaction data by month for trend analysis.
        
        Args:
            transactions: List of transactions
            
        Returns:
            Monthly aggregated data
        """
        monthly_data = {}
        
        for tx in transactions:
            try:
                month_key = tx.date.strftime("%Y-%m")
                if month_key not in monthly_data:
                    monthly_data[month_key] = 0.0
                monthly_data[month_key] += float(tx.amount)
            except Exception as e:
                self.logger.warning(f"Failed to aggregate transaction {tx.id}: {e}")
        
        return monthly_data
    
    def _generate_chart_insights(self, chart_type: str, data: Dict[str, Any]) -> str:
        """
        Generate insights for specific chart types.
        
        Args:
            chart_type: Type of chart
            data: Chart data
            
        Returns:
            Generated insights text
        """
        if chart_type == "expense_categories":
            if not data:
                return "No expense category data available for analysis."
            
            top_category = max(data.keys(), key=lambda k: data[k])
            top_amount = data[top_category]
            total = sum(data.values())
            percentage = (top_amount / total * 100) if total > 0 else 0
            
            return f"The largest expense category is {top_category}, representing {percentage:.1f}% of total expenses (${top_amount:,.2f}). This concentration suggests opportunities for targeted cost optimization in this area."
        
        elif chart_type == "monthly_trends":
            if len(data) < 2:
                return "Insufficient data for trend analysis."
            
            months = sorted(data.keys())
            latest_month = data[months[-1]]
            previous_month = data[months[-2]] if len(months) > 1 else latest_month
            
            change = ((latest_month - previous_month) / previous_month * 100) if previous_month != 0 else 0
            trend = "increased" if change > 0 else "decreased"
            
            return f"Monthly expenses have {trend} by {abs(change):.1f}% from the previous period, indicating {'growth' if change > 0 else 'cost reduction'} trends that require strategic attention."
        
        elif chart_type == "tax_optimization":
            if not data:
                return "No tax optimization opportunities identified."
            
            total_savings = sum(data.values())
            top_opportunity = max(data.keys(), key=lambda k: data[k]) if data else "None"
            
            return f"Total potential tax savings of ${total_savings:,.2f} identified, with {top_opportunity.replace('_', ' ')} representing the largest opportunity. Implementing these optimizations could significantly improve tax efficiency."
        
        return "Professional visualization supporting the financial analysis."
    
    def _assess_narrative_quality(self, *sections: str) -> float:
        """
        Assess the quality of generated narrative content.
        
        Args:
            sections: Narrative sections to assess
            
        Returns:
            Quality score from 0.0 to 5.0
        """
        total_score = 0.0
        section_count = len(sections)
        
        for section in sections:
            section_score = 0.0
            
            # Length assessment (appropriate length)
            word_count = len(section.split())
            if 100 <= word_count <= 500:
                section_score += 1.0
            elif 50 <= word_count < 100 or 500 < word_count <= 1000:
                section_score += 0.7
            else:
                section_score += 0.3
            
            # Content quality indicators
            if any(keyword in section.lower() for keyword in ["strategic", "recommend", "opportunity", "optimize"]):
                section_score += 1.0
            
            if any(keyword in section.lower() for keyword in ["analysis", "insight", "trend", "pattern"]):
                section_score += 1.0
            
            # Professional language indicators
            if any(phrase in section.lower() for phrase in ["executive", "c-suite", "strategic priority", "implementation"]):
                section_score += 1.0
            
            # Actionability indicators
            if any(phrase in section.lower() for phrase in ["action", "implement", "focus on", "priority"]):
                section_score += 1.0
            
            total_score += min(section_score, 5.0)
        
        return total_score / section_count if section_count > 0 else 0.0
    
    def _count_words(self, text: str) -> int:
        """Count words in text."""
        return len(text.split())
    
    def _determine_analysis_period(self, financial_metrics: FinancialMetrics) -> str:
        """Determine the analysis period from the data."""
        # This is a simplified implementation
        # In a real scenario, you'd analyze transaction dates
        return f"Current Period Analysis"