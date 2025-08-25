# Phase 3: Advanced Analytics & Report Generation

## Duration: 7-10 days
## Goal: Implement sophisticated analysis logic and consulting-grade report generation

### 3.1 Advanced Data Processing Implementation

**Enhance `src/utils/financial_calculations.py` with comprehensive analytics:**

```python
from typing import List, Dict
from decimal import Decimal
import statistics
import pandas as pd
from datetime import datetime, timedelta
from src.workflows.state_schemas import Transaction, FinancialMetrics
from src.utils.supabase_client import supabase_client
from src.utils.logging import StructuredLogger
from src.config.settings import settings

logger = StructuredLogger()

def calculate_financial_metrics(transactions: List[Transaction], client_id: str = None) -> FinancialMetrics:
    """Calculate comprehensive financial metrics with precision and store insights in Supabase"""
    
    # Separate revenue and expenses
    revenue_transactions = [t for t in transactions if t.amount > 0]
    expense_transactions = [t for t in transactions if t.amount < 0]
    
    # Core financial calculations
    total_revenue = sum(Decimal(str(t.amount)) for t in revenue_transactions)
    total_expenses = abs(sum(Decimal(str(t.amount)) for t in expense_transactions))
    gross_profit = total_revenue - total_expenses
    gross_margin_pct = float((gross_profit / total_revenue * 100)) if total_revenue > 0 else 0.0
    
    # Category analysis
    expense_by_category = categorize_expenses(expense_transactions)
    
    # Pattern recognition
    patterns = detect_business_patterns(transactions)
    
    # Anomaly detection
    anomalies = detect_statistical_anomalies(transactions)
    
    # Cash flow analysis
    cash_balance = calculate_cash_balance(transactions)
    
    # Store metrics in Supabase for historical analysis
    if client_id:
        try:
            supabase_client.table("financial_metrics").insert({
                "client_id": client_id,
                "calculation_date": datetime.now().isoformat(),
                "total_revenue": float(total_revenue),
                "total_expenses": float(total_expenses),
                "gross_profit": float(gross_profit),
                "gross_margin_pct": gross_margin_pct,
                "expense_categories": expense_by_category,
                "business_patterns": patterns["matches"],
                "detected_business_types": patterns["business_types"],
                "anomaly_count": len(anomalies)
            }).execute()
        except Exception as e:
            logger.log_agent_activity(
                "financial_metrics", "storage_failed", client_id,
                error=str(e)
            )
    
    return FinancialMetrics(
        total_revenue=total_revenue,
        total_expenses=total_expenses,
        gross_profit=gross_profit,
        gross_margin_pct=gross_margin_pct,
        cash_balance=cash_balance,
        expense_by_category=expense_by_category,
        anomalies=anomalies,
        pattern_matches=patterns["matches"],
        detected_business_types=patterns["business_types"]
    )

def detect_business_patterns(transactions: List[Transaction]) -> Dict:
    """Detect business patterns and trends"""
    patterns = {}
    
    # Seasonal analysis
    patterns["seasonal_variations"] = detect_seasonal_patterns(transactions)
    
    # Vendor concentration
    patterns["vendor_concentration"] = analyze_vendor_concentration(transactions)
    
    # Payment timing patterns
    patterns["payment_patterns"] = analyze_payment_timing(transactions)
    
    # Growth trends
    patterns["growth_trends"] = calculate_growth_trends(transactions)
    
    # Infer business types
    business_types = infer_business_types(transactions)
    
    return {
        "matches": patterns,
        "business_types": business_types[:3]  # Top 3 most likely
    }

def detect_statistical_anomalies(transactions: List[Transaction]) -> List[Transaction]:
    """Detect anomalies using statistical methods"""
    if len(transactions) < 2:
        return []
    
    amounts = [abs(float(t.amount)) for t in transactions]
    mean_amount = statistics.mean(amounts)
    std_amount = statistics.stdev(amounts) if len(amounts) > 1 else 0
    
    anomalies = []
    for transaction in transactions:
        if std_amount > 0:
            z_score = abs((abs(float(transaction.amount)) - mean_amount) / std_amount)
            if z_score > 3:  # 3-sigma rule
                anomalies.append(transaction)
    
    return anomalies

def analyze_vendor_concentration(transactions: List[Transaction]) -> Dict:
    """Analyze vendor concentration and dependency risk"""
    vendor_amounts = {}
    
    for transaction in transactions:
        if transaction.amount < 0:  # Expenses only
            vendor = extract_vendor_name(transaction.description)
            vendor_amounts[vendor] = vendor_amounts.get(vendor, 0) + abs(float(transaction.amount))
    
    total_expenses = sum(vendor_amounts.values())
    top_vendors = sorted(vendor_amounts.items(), key=lambda x: x[1], reverse=True)[:5]
    
    # Calculate concentration ratio (top 3 vendors)
    top_3_concentration = sum(amount for _, amount in top_vendors[:3]) / total_expenses if total_expenses > 0 else 0
    
    return {
        "top_vendors": top_vendors,
        "concentration_ratio": top_3_concentration,
        "risk_level": "high" if top_3_concentration > 0.6 else "medium" if top_3_concentration > 0.4 else "low"
    }

def calculate_growth_trends(transactions: List[Transaction]) -> Dict:
    """Calculate month-over-month and quarter-over-quarter growth"""
    # Group transactions by month
    monthly_data = {}
    
    for transaction in transactions:
        try:
            date = datetime.fromisoformat(transaction.date)
            month_key = date.strftime("%Y-%m")
            
            if month_key not in monthly_data:
                monthly_data[month_key] = {"revenue": 0, "expenses": 0}
            
            if transaction.amount > 0:
                monthly_data[month_key]["revenue"] += float(transaction.amount)
            else:
                monthly_data[month_key]["expenses"] += abs(float(transaction.amount))
        except:
            continue
    
    # Calculate growth rates
    months = sorted(monthly_data.keys())
    growth_rates = []
    
    for i in range(1, len(months)):
        prev_revenue = monthly_data[months[i-1]]["revenue"]
        curr_revenue = monthly_data[months[i]]["revenue"]
        
        if prev_revenue > 0:
            growth_rate = (curr_revenue - prev_revenue) / prev_revenue * 100
            growth_rates.append(growth_rate)
    
    avg_growth_rate = statistics.mean(growth_rates) if growth_rates else 0
    
    return {
        "monthly_data": monthly_data,
        "average_growth_rate": avg_growth_rate,
        "growth_trend": "increasing" if avg_growth_rate > 5 else "stable" if avg_growth_rate > -5 else "decreasing"
    }
```

### 3.2 Enhanced Tax Categorization

Before we do this, we need to find US IRS docuemnts related to taxation for the current year and ingest it into our RAG. We need to ensure that we have a good understanding of the taxation rules before implementing any changes. This needs to be done for the current year (2025), which includes the OBBB. The taxs categories below are rudimentary, but can be expanded upon based on the specific requirements of the use case. We should also enable the LLM to assign tax codes based off the information learned from the RAG. We need to locate the NAICS codes as well, and use them.

**Implement comprehensive tax rules in `src/agents/tax_categorizer.py`:**

```python
from typing import Dict, List
from decimal import Decimal
from src.workflows.state_schemas import Transaction, TaxSummary

# Comprehensive tax category mapping
TAX_CATEGORIES = {
    "travel": {
        "keywords": ["hotel", "flight", "uber", "taxi", "mileage", "car rental", "airbnb"],
        "deductible": True,
        "percentage": 1.0
    },
    "meals": {
        "keywords": ["restaurant", "catering", "lunch", "dinner", "food", "coffee"],
        "deductible": True,
        "percentage": 0.5  # 50% deductible for business meals
    },
    "office_supplies": {
        "keywords": ["staples", "office depot", "supplies", "paper", "pens"],
        "deductible": True,
        "percentage": 1.0
    },
    "professional_services": {
        "keywords": ["legal", "accounting", "consulting", "attorney", "cpa"],
        "deductible": True,
        "percentage": 1.0
    },
    "marketing": {
        "keywords": ["advertising", "promotion", "social media", "facebook ads", "google ads"],
        "deductible": True,
        "percentage": 1.0
    },
    "utilities": {
        "keywords": ["electric", "gas", "internet", "phone", "water", "utilities"],
        "deductible": True,
        "percentage": 1.0
    },
    "equipment": {
        "keywords": ["computer", "software", "hardware", "laptop", "printer"],
        "deductible": True,
        "percentage": 1.0
    },
    "insurance": {
        "keywords": ["liability", "business insurance", "workers comp"],
        "deductible": True,
        "percentage": 1.0
    },
    "rent": {
        "keywords": ["office rent", "warehouse", "commercial rent"],
        "deductible": True,
        "percentage": 1.0
    },
    "personal": {
        "keywords": ["personal", "salary", "owner draw"],
        "deductible": False,
        "percentage": 0.0
    }
}

def categorize_transaction(transaction: Transaction) -> str:
    """Categorize transaction using rule-based approach"""
    description = transaction.description.lower()
    
    # Score each category
    category_scores = {}
    for category, config in TAX_CATEGORIES.items():
        score = 0
        for keyword in config["keywords"]:
            if keyword in description:
                score += len(keyword)  # Longer keywords get higher scores
        category_scores[category] = score
    
    # Return highest scoring category
    best_category = max(category_scores.items(), key=lambda x: x[1])
    return best_category[0] if best_category[1] > 0 else "other"

def calculate_deductible_amount(transaction: Transaction, category: str) -> Decimal:
    """Calculate deductible amount based on category rules"""
    if category not in TAX_CATEGORIES:
        return Decimal('0')
    
    config = TAX_CATEGORIES[category]
    if not config["deductible"]:
        return Decimal('0')
    
    amount = abs(Decimal(str(transaction.amount)))
    percentage = Decimal(str(config["percentage"]))
    
    return amount * percentage

def calculate_tax_summary(transactions: List[Transaction]) -> TaxSummary:
    """Calculate comprehensive tax summary"""
    deductible_total = Decimal('0')
    non_deductible_total = Decimal('0')
    flags = []
    
    category_totals = {}
    
    for transaction in transactions:
        if transaction.amount >= 0:  # Skip income transactions
            continue
            
        category = transaction.tax_category or "other"
        deductible_amount = calculate_deductible_amount(transaction, category)
        
        if deductible_amount > 0:
            deductible_total += deductible_amount
            transaction.is_deductible = True
        else:
            non_deductible_total += abs(Decimal(str(transaction.amount)))
            transaction.is_deductible = False
        
        # Track by category
        if category not in category_totals:
            category_totals[category] = Decimal('0')
        category_totals[category] += abs(Decimal(str(transaction.amount)))
    
    # Generate flags based on analysis
    if deductible_total > Decimal('50000'):
        flags.append("High deductible amount may require additional documentation")
    
    # Check for unusual patterns
    meals_total = category_totals.get("meals", Decimal('0'))
    if meals_total > deductible_total * Decimal('0.3'):
        flags.append("Meals expenses exceed 30% of total - review for reasonableness")
    
    travel_total = category_totals.get("travel", Decimal('0'))
    if travel_total > deductible_total * Decimal('0.4'):
        flags.append("Travel expenses are significant - ensure proper documentation")
    
    # Calculate potential tax savings (estimated 25% tax rate)
    potential_savings = deductible_total * Decimal('0.25')
    
    return TaxSummary(
        deductible_total=deductible_total,
        non_deductible_total=non_deductible_total,
        potential_savings=potential_savings,
        flags=flags,
        categorization_accuracy=95.0  # Based on rule confidence
    )
```

### 3.3 LLM Integration & Report Generation

**Implement sophisticated report generation in `src/agents/report_generator.py`:**

```python
from typing import Dict, Any, List
import openai
from src.workflows.state_schemas import OverallState
from src.config.settings import settings
from src.config.personas import fill_persona_placeholders
from src.utils.supabase_client import supabase_client
from src.utils.logging import StructuredLogger
import tempfile
import os

logger = StructuredLogger()

# Comprehensive system prompt for consulting-grade reports
SYSTEM_PROMPT_TEMPLATE = """
You are Professor Elena Castellanos, Executive Financial Storytelling Director with 15+ years at McKinsey & Company. You synthesize financial intelligence into compelling C-suite narratives.

TEAM ANALYSIS CREDITS:
- Dr. Victoria Ashworth (Chief Financial Operations Orchestrator): {orchestrator_achievement}
- Dr. Marcus Thornfield (Senior Market Intelligence Analyst): {data_fetcher_achievement}
- Alexandra Sterling (Chief Data Transformation Specialist): {data_cleaner_achievement}
- Dexter Blackwood (Quantitative Data Integrity Analyst): {data_processor_achievement}
- Clarke Pemberton (Corporate Tax Compliance Strategist): {tax_categorizer_achievement}

FINANCIAL DATA SUMMARY:
{financial_data_summary}

REPORT STRUCTURE REQUIRED:
1. **Executive Summary** - Key findings, critical decisions needed, and strategic implications
2. **Financial Performance Overview** - Revenue, expenses, profitability analysis with trends
3. **Business Intelligence Analysis** - Patterns, vendor relationships, operational insights
4. **Risk Assessment** - Anomalies, concentration risks, volatility analysis
5. **Tax Optimization Analysis** - Deductions, compliance status, savings opportunities
6. **Strategic Recommendations** - Actionable next steps with implementation timeline
7. **Market Intelligence & Regional Context** - Industry benchmarks and economic factors
8. **Conclusion & Expected Outcomes** - Projected impact of implementing recommendations

WRITING REQUIREMENTS:
- Use McKinsey-level analytical rigor with data-driven insights
- Include specific numbers and percentages from the analysis
- Professional yet accessible language for C-suite executives
- Reference charts where applicable: "(See Figure X)"
- Provide implementation timelines for recommendations
- Ground all insights in actual data - avoid speculation
- Use section headers exactly as specified above
- Target 2000-3000 words for comprehensive analysis

Generate a consulting-grade financial intelligence report following this structure precisely.
"""

def generate_narrative_report(state: OverallState) -> Dict[str, Any]:
    """Generate consulting-grade narrative using LLM and store in Supabase"""
    trace_id = state.get("trace_id", "unknown")
    client_id = state.get("client_id")
    
    try:
        # Prepare financial data summary
        data_summary = prepare_financial_data_summary(state)
        
        # Fill persona placeholders
        persona_achievements = fill_persona_placeholders(state)
        
        # Construct prompt
        prompt = SYSTEM_PROMPT_TEMPLATE.format(
            **persona_achievements,
            financial_data_summary=data_summary
        )
        
        # Generate report with model routing
        report_content = call_llm_with_retry(prompt, max_retries=3)
        
        # Store report content in Supabase Storage
        storage_path = None
        if client_id:
            storage_path = store_report_in_supabase(report_content, client_id, trace_id)
        
        return {
            "report_content": report_content,
            "storage_path": storage_path,
            "success": True
        }
        
    except Exception as e:
        logger.log_agent_activity(
            "report_generator", "generation_failed", trace_id,
            error=str(e)
        )
        return {
            "report_content": None,
            "storage_path": None,
            "success": False,
            "error": str(e)
        }

def store_report_in_supabase(report_content: str, client_id: str, trace_id: str) -> str:
    """Store report content in Supabase Storage"""
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as tmp_file:
            tmp_file.write(report_content)
            tmp_file_path = tmp_file.name
        
        # Upload to Supabase storage
        with open(tmp_file_path, 'rb') as file:
            storage_path = f"{client_id}/{trace_id}/narrative_report.md"
            
            result = supabase_client.storage.from_(settings.storage_bucket).upload(
                storage_path, file
            )
            
            if result.get('error'):
                raise Exception(f"Storage upload failed: {result['error']}")
        
        # Clean up temp file
        os.unlink(tmp_file_path)
        
        return storage_path
        
    except Exception as e:
        logger.log_agent_activity(
            "report_storage", "upload_failed", trace_id,
            error=str(e)
        )
        raise

def prepare_financial_data_summary(state: OverallState) -> str:
    """Prepare structured data summary for LLM input"""
    
    transactions = state.get("transactions", [])
    financial_metrics = state.get("financial_metrics")
    tax_summary = state.get("tax_summary")
    
    summary_parts = []
    
    # Transaction overview
    summary_parts.append(f"TRANSACTION OVERVIEW:")
    summary_parts.append(f"- Total transactions analyzed: {len(transactions)}")
    summary_parts.append(f"- Date range: {get_date_range(transactions)}")
    
    # Financial metrics
    if financial_metrics:
        summary_parts.append(f"\nFINANCIAL METRICS:")
        summary_parts.append(f"- Total Revenue: ${financial_metrics.total_revenue:,.2f}")
        summary_parts.append(f"- Total Expenses: ${financial_metrics.total_expenses:,.2f}")
        summary_parts.append(f"- Gross Profit: ${financial_metrics.gross_profit:,.2f}")
        summary_parts.append(f"- Gross Margin: {financial_metrics.gross_margin_pct:.1f}%")
        
        # Expense breakdown
        if financial_metrics.expense_by_category:
            summary_parts.append(f"\nEXPENSE CATEGORIES:")
            for category, amount in financial_metrics.expense_by_category.items():
                summary_parts.append(f"- {category.title()}: ${amount:,.2f}")
        
        # Business patterns
        if financial_metrics.detected_business_types:
            summary_parts.append(f"\nBUSINESS PATTERNS:")
            for business_type in financial_metrics.detected_business_types:
                summary_parts.append(f"- {business_type}")
        
        # Anomalies
        if financial_metrics.anomalies:
            summary_parts.append(f"\nANOMALIES DETECTED:")
            for anomaly in financial_metrics.anomalies[:3]:  # Top 3
                summary_parts.append(f"- ${abs(float(anomaly.amount)):,.2f}: {anomaly.description}")
    
    # Tax analysis
    if tax_summary:
        summary_parts.append(f"\nTAX ANALYSIS:")
        summary_parts.append(f"- Deductible Expenses: ${tax_summary.deductible_total:,.2f}")
        summary_parts.append(f"- Non-Deductible: ${tax_summary.non_deductible_total:,.2f}")
        summary_parts.append(f"- Estimated Tax Savings: ${tax_summary.potential_savings:,.2f}")
        
        if tax_summary.flags:
            summary_parts.append(f"\nTAX COMPLIANCE FLAGS:")
            for flag in tax_summary.flags:
                summary_parts.append(f"- {flag}")
    
    return "\n".join(summary_parts)

def call_llm_with_retry(prompt: str, max_retries: int = 3) -> str:
    """Call LLM with error handling and retries"""
    
    for attempt in range(max_retries):
        try:
            # Model selection based on prompt length
            model = choose_model_for_task("report_generation", len(prompt))
            
            if settings.llm_provider == "openai":
                client = openai.OpenAI(api_key=settings.openai_api_key)
                
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=4000
                )
                
                return response.choices[0].message.content
                
            elif settings.llm_provider == "local":
                return call_local_llm(prompt)
            
            else:
                raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")
                
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            continue
    
    raise Exception("Failed to generate report after all retries")

def choose_model_for_task(task_type: str, content_length: int) -> str:
    """Select appropriate model based on task requirements"""
    
    if task_type == "report_generation":
        if content_length > 50000:
            return "gpt-4.1"  # Better context handling
        else:
            return "gpt-4.1"  # Latest model for quality
    
    return "gpt-4.1"  # Default
```

### 3.4 Chart Generation

**Implement visualization generation in `src/utils/chart_generation.py`:**

```python
from pyecharts.charts import Line, Bar, Pie
from pyecharts import options as opts
from pyecharts.globals import ThemeType
from pyecharts.render import make_snapshot
from snapshot_selenium import snapshot
import pandas as pd
from typing import List, Dict
import os
from decimal import Decimal
from src.utils.supabase_client import supabase_client
from src.config.settings import settings
from src.utils.logging import StructuredLogger

logger = StructuredLogger()

def generate_visualizations(financial_metrics, client_id: str = None, trace_id: str = None, output_dir: str = "/tmp/charts") -> List[str]:
    """Generate charts for financial analysis using Apache ECharts and store in Supabase Storage"""
    
    os.makedirs(output_dir, exist_ok=True)
    chart_files = []
    supabase_chart_paths = []
    
    try:
        # 1. Expense by Category Pie Chart
        if financial_metrics and financial_metrics.expense_by_category:
            chart_path = generate_expense_pie_chart(
                financial_metrics.expense_by_category, 
                output_dir
            )
            chart_files.append(chart_path)
            
            # Upload to Supabase Storage
            if client_id and trace_id:
                supabase_path = upload_chart_to_supabase(chart_path, client_id, trace_id, "expense_categories_pie.png")
                if supabase_path:
                    supabase_chart_paths.append(supabase_path)
        
        # 2. Revenue vs Expenses Bar Chart
        if financial_metrics:
            chart_path = generate_revenue_expense_chart(
                financial_metrics,
                output_dir
            )
            chart_files.append(chart_path)
            
            # Upload to Supabase Storage
            if client_id and trace_id:
                supabase_path = upload_chart_to_supabase(chart_path, client_id, trace_id, "revenue_expense_overview.png")
                if supabase_path:
                    supabase_chart_paths.append(supabase_path)
    
    except Exception as e:
        logger.log_agent_activity(
            "chart_generator", "generation_failed", trace_id,
            error=str(e)
        )
    
    # Return Supabase paths if available, otherwise local paths
    return supabase_chart_paths if supabase_chart_paths else chart_files

def upload_chart_to_supabase(local_path: str, client_id: str, trace_id: str, filename: str) -> str:
    """Upload chart to Supabase Storage"""
    try:
        with open(local_path, 'rb') as file:
            storage_path = f"{client_id}/{trace_id}/charts/{filename}"
            
            result = supabase_client.storage.from_(settings.charts_bucket).upload(
                storage_path, file
            )
            
            if result.get('error'):
                raise Exception(f"Chart upload failed: {result['error']}")
            
            return storage_path
            
    except Exception as e:
        logger.log_agent_activity(
            "chart_upload", "upload_failed", trace_id,
            error=str(e)
        )
        return None

def generate_expense_pie_chart(expense_by_category: Dict[str, Decimal], output_dir: str) -> str:
    """Generate pie chart for expense categories using Apache ECharts"""
    
    # Prepare data for pyecharts
    data_pairs = [(category, float(amount)) for category, amount in expense_by_category.items()]
    
    # Create pie chart with Apache ECharts
    pie_chart = (
        Pie(init_opts=opts.InitOpts(theme=ThemeType.WESTEROS, width="800px", height="600px"))
        .add(
            series_name="Expenses",
            data_pair=data_pairs,
            radius=["40%", "75%"],
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(
                title="Expenses by Category",
                pos_left="center",
                title_textstyle_opts=opts.TextStyleOpts(font_size=20, font_weight="bold")
            ),
            legend_opts=opts.LegendOpts(orient="vertical", pos_top="15%", pos_left="2%"),
            toolbox_opts=opts.ToolboxOpts(),
        )
        .set_series_opts(
            tooltip_opts=opts.TooltipOpts(
                trigger="item", formatter="{a} <br/>{b}: ${c} ({d}%)"
            ),
            label_opts=opts.LabelOpts(formatter="{b}: {d}%"),
        )
    )
    
    # Save chart as PNG
    chart_path = os.path.join(output_dir, "expense_categories_pie.png")
    make_snapshot(snapshot, pie_chart.render(), chart_path)
    
    return chart_path

def generate_revenue_expense_chart(financial_metrics, output_dir: str) -> str:
    """Generate bar chart comparing revenue and expenses using Apache ECharts"""
    
    categories = ['Revenue', 'Expenses', 'Profit']
    values = [
        float(financial_metrics.total_revenue),
        float(financial_metrics.total_expenses),
        float(financial_metrics.gross_profit)
    ]
    
    # Create bar chart with Apache ECharts
    bar_chart = (
        Bar(init_opts=opts.InitOpts(theme=ThemeType.WESTEROS, width="800px", height="600px"))
        .add_xaxis(categories)
        .add_yaxis(
            series_name="Amount ($)",
            y_axis=values,
            itemstyle_opts=opts.ItemStyleOpts(
                color_function=opts.JsCode(
                    "function(params) {"
                    "var colors = ['#5470c6', '#ee6666', '#73c0de'];"
                    "return colors[params.dataIndex];"
                    "}"
                )
            )
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(
                title="Financial Overview",
                pos_left="center",
                title_textstyle_opts=opts.TextStyleOpts(font_size=20, font_weight="bold")
            ),
            xaxis_opts=opts.AxisOpts(name="Category"),
            yaxis_opts=opts.AxisOpts(
                name="Amount ($)",
                axislabel_opts=opts.LabelOpts(formatter="${value}")
            ),
            toolbox_opts=opts.ToolboxOpts(),
            datazoom_opts=[opts.DataZoomOpts()],
        )
        .set_series_opts(
            tooltip_opts=opts.TooltipOpts(
                trigger="axis",
                axis_pointer_type="cross",
                formatter=opts.JsCode(
                    "function(params) {"
                    "var value = params[0].value;"
                    "return params[0].name + ': $' + value.toLocaleString();"
                    "}"
                )
            ),
            label_opts=opts.LabelOpts(
                is_show=True,
                formatter=opts.JsCode(
                    "function(params) { return '$' + params.value.toLocaleString(); }"
                )
            ),
        )
    )
    
    # Save chart as PNG
    chart_path = os.path.join(output_dir, "revenue_expense_overview.png")
    make_snapshot(snapshot, bar_chart.render(), chart_path)
    
    return chart_path
```

### 3.5 PDF Generation

**Implement PDF conversion in `src/utils/pdf_generation.py`:**

```python
import markdown
import weasyprint
from typing import List
import os
import tempfile
from src.utils.supabase_client import supabase_client
from src.config.settings import settings
from src.utils.logging import StructuredLogger

logger = StructuredLogger()

def convert_to_pdf(markdown_content: str, charts: List[str] = None, client_id: str = None, trace_id: str = None) -> Dict[str, str]:
    """Convert markdown report to PDF with embedded charts and store in Supabase"""
    
    try:
        # Insert chart references into markdown
        if charts:
            chart_html = generate_chart_html(charts)
            markdown_content = embed_charts_in_markdown(markdown_content, chart_html)
        
        # Convert markdown to HTML
        html_content = markdown.markdown(markdown_content, extensions=['tables', 'toc'])
        
        # Add CSS styling
        styled_html = add_pdf_styling(html_content)
        
        # Generate PDF locally
        local_pdf_path = generate_pdf_from_html(styled_html)
        
        # Upload to Supabase Storage
        supabase_path = None
        if client_id and trace_id:
            supabase_path = upload_pdf_to_supabase(local_pdf_path, client_id, trace_id)
        
        return {
            "local_path": local_pdf_path,
            "supabase_path": supabase_path,
            "success": True
        }
        
    except Exception as e:
        logger.log_agent_activity(
            "pdf_generator", "conversion_failed", trace_id,
            error=str(e)
        )
        return {
            "local_path": None,
            "supabase_path": None,
            "success": False,
            "error": str(e)
        }

def upload_pdf_to_supabase(local_path: str, client_id: str, trace_id: str) -> str:
    """Upload PDF to Supabase Storage"""
    try:
        with open(local_path, 'rb') as file:
            storage_path = f"{client_id}/{trace_id}/final_report.pdf"
            
            result = supabase_client.storage.from_(settings.storage_bucket).upload(
                storage_path, file
            )
            
            if result.get('error'):
                raise Exception(f"PDF upload failed: {result['error']}")
            
            # Clean up local file
            os.unlink(local_path)
            
            return storage_path
            
    except Exception as e:
        logger.log_agent_activity(
            "pdf_upload", "upload_failed", trace_id,
            error=str(e)
        )
        return None

def add_pdf_styling(html_content: str) -> str:
    """Add professional styling for PDF generation"""
    
    css_styles = """
    <style>
    body {
        font-family: Arial, sans-serif;
        line-height: 1.6;
        color: #333;
        max-width: 800px;
        margin: 0 auto;
        padding: 20px;
    }
    h1 { color: #2c3e50; border-bottom: 2px solid #3498db; }
    h2 { color: #34495e; margin-top: 30px; }
    h3 { color: #7f8c8d; }
    .executive-summary { background: #ecf0f1; padding: 15px; border-left: 4px solid #3498db; }
    table { border-collapse: collapse; width: 100%; margin: 20px 0; }
    th, td { border: 1px solid #bdc3c7; padding: 8px; text-align: left; }
    th { background-color: #34495e; color: white; }
    .chart { text-align: center; margin: 20px 0; }
    </style>
    """
    
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Financial Analysis Report</title>
        {css_styles}
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """

def generate_pdf_from_html(html_content: str) -> str:
    """Generate PDF from HTML using WeasyPrint"""
    
    # Create temporary file for PDF
    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
        pdf_path = tmp_file.name
    
    # Generate PDF
    weasyprint.HTML(string=html_content).write_pdf(pdf_path)
    
    return pdf_path
```

### 3.6 Data Quality Enforcement

**Implement comprehensive validation in `src/utils/data_validation.py`:**

```python
from typing import List, Dict, Tuple
from decimal import Decimal, InvalidOperation
from datetime import datetime
from src.workflows.state_schemas import Transaction
from src.utils.supabase_client import supabase_client
from src.utils.logging import StructuredLogger

logger = StructuredLogger()

def validate_transactions(transactions: List[Transaction], client_id: str = None) -> Tuple[List[Transaction], List[str]]:
    """Validate and clean transaction data with quality metrics stored in Supabase"""
    
    validated_transactions = []
    warnings = []
    
    for i, transaction in enumerate(transactions):
        validation_result = validate_single_transaction(transaction, i)
        
        if validation_result["is_valid"]:
            validated_transactions.append(validation_result["transaction"])
        else:
            warnings.extend(validation_result["warnings"])
    
    # Perform cross-validation
    cross_validation_warnings = cross_validate_transactions(validated_transactions)
    warnings.extend(cross_validation_warnings)
    
    # Store data quality metrics in Supabase
    if client_id:
        try:
            quality_score = (len(validated_transactions) / len(transactions) * 100) if transactions else 0
            
            supabase_client.table("data_quality_metrics").insert({
                "client_id": client_id,
                "validation_date": datetime.now().isoformat(),
                "total_transactions": len(transactions),
                "valid_transactions": len(validated_transactions),
                "quality_score": quality_score,
                "warning_count": len(warnings),
                "warnings": warnings[:10]  # Store first 10 warnings
            }).execute()
        except Exception as e:
            logger.log_agent_activity(
                "data_validation", "metrics_storage_failed", client_id,
                error=str(e)
            )
    
    return validated_transactions, warnings

def validate_single_transaction(transaction: Transaction, index: int) -> Dict:
    """Validate individual transaction"""
    
    warnings = []
    is_valid = True
    
    # Validate date
    try:
        datetime.fromisoformat(transaction.date)
    except ValueError:
        warnings.append(f"Transaction {index}: Invalid date format '{transaction.date}'")
        is_valid = False
    
    # Validate amount
    try:
        amount = Decimal(str(transaction.amount))
        if amount == 0:
            warnings.append(f"Transaction {index}: Zero amount transaction")
    except (InvalidOperation, ValueError):
        warnings.append(f"Transaction {index}: Invalid amount '{transaction.amount}'")
        is_valid = False
    
    # Validate description
    if not transaction.description or len(transaction.description.strip()) == 0:
        warnings.append(f"Transaction {index}: Empty description")
        is_valid = False
    
    return {
        "is_valid": is_valid,
        "transaction": transaction,
        "warnings": warnings
    }

def cross_validate_transactions(transactions: List[Transaction]) -> List[str]:
    """Perform cross-validation checks across all transactions"""
    
    warnings = []
    
    if len(transactions) == 0:
        warnings.append("No valid transactions found")
        return warnings
    
    # Check for duplicate transactions
    duplicates = find_duplicate_transactions(transactions)
    if duplicates:
        warnings.append(f"Found {len(duplicates)} potential duplicate transactions")
    
    # Check date range consistency
    date_warnings = validate_date_range(transactions)
    warnings.extend(date_warnings)
    
    # Check for unusual patterns
    pattern_warnings = detect_unusual_patterns(transactions)
    warnings.extend(pattern_warnings)
    
    return warnings
```

### 3.7 Vector Database Integration for Analytics Insights

**Implement analytics insights storage and retrieval using Supabase pgvector in `src/utils/analytics_vector_ops.py`:**

```python
from typing import List, Dict, Any, Optional
import numpy as np
from src.utils.supabase_client import supabase_client
from src.config.settings import settings
from src.utils.logging import StructuredLogger
import openai

logger = StructuredLogger()

def store_analysis_insights(client_id: str, analysis_type: str, insights: Dict[str, Any], trace_id: str) -> bool:
    """Store analysis insights as embeddings for future pattern matching"""
    try:
        # Create text representation of insights
        insight_text = format_insights_for_embedding(insights)
        
        # Generate embedding
        embedding = generate_embedding(insight_text)
        
        # Store in vector database
        result = supabase_client.table("analytics_insights").insert({
            "client_id": client_id,
            "trace_id": trace_id,
            "analysis_type": analysis_type,
            "insight_text": insight_text,
            "embedding": embedding,
            "metadata": insights,
            "created_at": "now()"
        }).execute()
        
        return True
        
    except Exception as e:
        logger.log_agent_activity(
            "vector_storage", "insight_storage_failed", trace_id,
            error=str(e)
        )
        return False

def find_similar_analyses(client_id: str, current_insights: Dict[str, Any], similarity_threshold: float = 0.8) -> List[Dict]:
    """Find similar past analyses using vector similarity search"""
    try:
        # Generate embedding for current insights
        insight_text = format_insights_for_embedding(current_insights)
        query_embedding = generate_embedding(insight_text)
        
        # Perform similarity search using pgvector
        # This uses Supabase's built-in vector similarity functions
        similar_analyses = supabase_client.rpc(
            "match_analytics_insights",
            {
                "query_embedding": query_embedding,
                "match_threshold": similarity_threshold,
                "match_count": 5,
                "client_id": client_id
            }
        ).execute()
        
        return similar_analyses.data if similar_analyses.data else []
        
    except Exception as e:
        logger.log_agent_activity(
            "vector_search", "similarity_search_failed", client_id,
            error=str(e)
        )
        return []

def format_insights_for_embedding(insights: Dict[str, Any]) -> str:
    """Format insights dictionary into text suitable for embedding"""
    text_parts = []
    
    # Format financial metrics
    if "financial_metrics" in insights:
        metrics = insights["financial_metrics"]
        text_parts.append(f"Revenue: ${metrics.get('total_revenue', 0):,.2f}")
        text_parts.append(f"Expenses: ${metrics.get('total_expenses', 0):,.2f}")
        text_parts.append(f"Profit Margin: {metrics.get('gross_margin_pct', 0):.1f}%")
    
    # Format business patterns
    if "business_patterns" in insights:
        patterns = insights["business_patterns"]
        if patterns.get("detected_business_types"):
            text_parts.append(f"Business types: {', '.join(patterns['detected_business_types'])}")
    
    # Format risk indicators
    if "risk_indicators" in insights:
        risk = insights["risk_indicators"]
        text_parts.append(f"Risk level: {risk.get('overall_risk', 'unknown')}")
    
    return " | ".join(text_parts)

def generate_embedding(text: str) -> List[float]:
    """Generate embedding for text using OpenAI"""
    client = openai.OpenAI(api_key=settings.openai_api_key)
    
    response = client.embeddings.create(
        model=settings.embedding_model,
        input=text
    )
    
    return response.data[0].embedding
```

**Create database function for similarity search (`supabase/migrations/006_analytics_similarity_function.sql`):**

```sql
-- Function to find similar analytics insights using cosine similarity
CREATE OR REPLACE FUNCTION match_analytics_insights(
  query_embedding vector(1536),
  match_threshold float,
  match_count int,
  client_id uuid
)
RETURNS TABLE (
  id uuid,
  trace_id text,
  analysis_type text,
  insight_text text,
  metadata jsonb,
  similarity float
)
LANGUAGE plpgsql
AS $$
BEGIN
  RETURN QUERY
  SELECT
    analytics_insights.id,
    analytics_insights.trace_id,
    analytics_insights.analysis_type,
    analytics_insights.insight_text,
    analytics_insights.metadata,
    1 - (analytics_insights.embedding <=> query_embedding) AS similarity
  FROM analytics_insights
  WHERE 
    analytics_insights.client_id = match_analytics_insights.client_id
    AND 1 - (analytics_insights.embedding <=> query_embedding) > match_threshold
  ORDER BY analytics_insights.embedding <=> query_embedding
  LIMIT match_count;
END;
$$;
```

### 3.8 Enhanced Analytics with Historical Context

**Implement historical trend analysis in `src/utils/historical_analytics.py`:**

```python
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from decimal import Decimal
from src.utils.supabase_client import supabase_client
from src.workflows.state_schemas import FinancialMetrics

def analyze_performance_trends(client_id: str, current_metrics: FinancialMetrics) -> Dict[str, Any]:
    """Analyze performance trends using historical data from Supabase"""
    try:
        # Get historical metrics from last 12 months
        historical_data = get_historical_metrics(client_id, months=12)
        
        if not historical_data:
            return {"trend_analysis": "insufficient_historical_data"}
        
        # Calculate trends
        revenue_trend = calculate_revenue_trend(historical_data, current_metrics)
        expense_trend = calculate_expense_trend(historical_data, current_metrics)
        profitability_trend = calculate_profitability_trend(historical_data, current_metrics)
        
        # Compare against industry benchmarks (if available)
        benchmark_comparison = compare_to_benchmarks(current_metrics)
        
        return {
            "revenue_trend": revenue_trend,
            "expense_trend": expense_trend,
            "profitability_trend": profitability_trend,
            "benchmark_comparison": benchmark_comparison,
            "trend_analysis": "comprehensive"
        }
        
    except Exception as e:
        return {"trend_analysis": "error", "error": str(e)}

def get_historical_metrics(client_id: str, months: int = 12) -> List[Dict]:
    """Retrieve historical financial metrics from Supabase"""
    cutoff_date = datetime.now() - timedelta(days=months * 30)
    
    result = supabase_client.table("financial_metrics").select("*").eq(
        "client_id", client_id
    ).gte(
        "calculation_date", cutoff_date.isoformat()
    ).order("calculation_date").execute()
    
    return result.data if result.data else []

def calculate_revenue_trend(historical_data: List[Dict], current_metrics: FinancialMetrics) -> Dict[str, Any]:
    """Calculate revenue growth trends"""
    if len(historical_data) < 2:
        return {"trend": "insufficient_data"}
    
    revenues = [float(record["total_revenue"]) for record in historical_data]
    current_revenue = float(current_metrics.total_revenue)
    
    # Calculate month-over-month growth
    recent_growth = ((current_revenue - revenues[-1]) / revenues[-1] * 100) if revenues[-1] > 0 else 0
    
    # Calculate average growth rate
    growth_rates = []
    for i in range(1, len(revenues)):
        if revenues[i-1] > 0:
            growth_rate = (revenues[i] - revenues[i-1]) / revenues[i-1] * 100
            growth_rates.append(growth_rate)
    
    avg_growth = sum(growth_rates) / len(growth_rates) if growth_rates else 0
    
    return {
        "trend": "increasing" if avg_growth > 5 else "stable" if avg_growth > -5 else "decreasing",
        "recent_growth_pct": recent_growth,
        "average_growth_pct": avg_growth,
        "volatility": calculate_volatility(growth_rates)
    }
```

## Phase 3 Acceptance Criteria

- [ ] System produces detailed narrative report for sample input
- [ ] Report includes all 8 required sections with coherent content
- [ ] All target analytics present (risk assessment, market insight, tax analysis)
- [ ] PDF successfully generated with at least one chart
- [ ] No critical data quality issues in processing
- [ ] Tax categorization achieves >95% accuracy on test data
- [ ] Financial calculations precise to the cent using Decimal
- [ ] LLM integration working with model routing
- [ ] Charts generated and embedded in reports
- [ ] **Supabase Storage integration working for reports and charts**
- [ ] **Vector database storing analytics insights with embeddings**
- [ ] **Historical trend analysis using Supabase-stored metrics**
- [ ] **Data quality metrics tracked and stored in Supabase**
- [ ] Data validation catches and reports quality issues
- [ ] Test coverage ≥85% for new analytics logic
- [ ] Performance acceptable for 1000+ transaction datasets
- [ ] **Similarity search finding relevant past analyses**
- [ ] **All file storage operations using Supabase Storage buckets**

## Next Steps

After Phase 3 completion, proceed to Phase 4: Testing, Hardening, and Containerization.

## RACI Matrix

**Responsible:** Solo Developer
**Accountable:** Solo Developer  
**Consulted:** AI assistant for coding, domain expert for financial logic validation
**Informed:** Stakeholder gets first full report output for feedback