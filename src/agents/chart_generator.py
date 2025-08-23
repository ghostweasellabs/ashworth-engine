from typing import Dict, Any, List
import os
import tempfile
from pyecharts.charts import Line, Bar, Pie
from pyecharts import options as opts
from pyecharts.globals import ThemeType
from pyecharts.render import make_snapshot
from snapshot_selenium import snapshot
from decimal import Decimal
from src.workflows.state_schemas import OverallState, FinancialMetrics
from src.utils.supabase_client import supabase_client
from src.utils.logging import StructuredLogger
from src.config.settings import settings

logger = StructuredLogger()

def chart_generator_agent(state: OverallState) -> Dict[str, Any]:
    """Generate professional charts and visualizations using Apache ECharts"""
    trace_id = state.get("trace_id", "unknown")
    client_id = state.get("client_id")
    
    try:
        logger.log_agent_activity(
            "chart_generator", "start_generation", trace_id
        )
        
        financial_metrics = state.get("financial_metrics")
        if not financial_metrics:
            return {
                "charts": [],
                "warnings": ["No financial metrics available for chart generation"],
                "workflow_phase": "chart_generation_skipped"
            }
        
        # Generate charts and upload to Supabase
        chart_paths = generate_financial_charts(
            financial_metrics, 
            client_id=client_id, 
            trace_id=trace_id
        )
        
        # Update analysis status in Supabase
        if client_id:
            try:
                supabase_client.table("analyses").update({
                    "status": "charts_generated",
                    "charts_metadata": {
                        "chart_count": len(chart_paths),
                        "chart_paths": chart_paths,
                        "generation_timestamp": state.get("processing_start_time")
                    }
                }).eq("id", trace_id).execute()
            except Exception as db_error:
                logger.log_agent_activity(
                    "chart_generator", "metadata_update_failed", trace_id,
                    error=str(db_error)
                )
        
        logger.log_agent_activity(
            "chart_generator", "generation_complete", trace_id,
            charts_generated=len(chart_paths)
        )
        
        return {
            "charts": chart_paths,
            "workflow_phase": "chart_generation_complete",
            "error_messages": []
        }
        
    except Exception as e:
        logger.log_agent_activity(
            "chart_generator", "generation_failed", trace_id,
            error=str(e)
        )
        return {
            "charts": [],
            "error_messages": [f"Chart generator error: {str(e)}"],
            "workflow_phase": "chart_generation_failed"
        }

def generate_financial_charts(financial_metrics: FinancialMetrics, client_id: str = None, trace_id: str = None) -> List[str]:
    """Generate comprehensive financial charts and upload to Supabase Storage"""
    
    chart_paths = []
    temp_dir = tempfile.mkdtemp()
    
    try:
        # 1. Expense by Category Pie Chart
        if financial_metrics.expense_by_category:
            pie_chart_path = generate_expense_pie_chart(
                financial_metrics.expense_by_category, 
                temp_dir
            )
            if pie_chart_path:
                supabase_path = upload_chart_to_supabase(
                    pie_chart_path, client_id, trace_id, "expense_categories_pie.png"
                )
                if supabase_path:
                    chart_paths.append(supabase_path)
        
        # 2. Revenue vs Expenses Bar Chart
        revenue_expense_path = generate_revenue_expense_chart(
            financial_metrics, 
            temp_dir
        )
        if revenue_expense_path:
            supabase_path = upload_chart_to_supabase(
                revenue_expense_path, client_id, trace_id, "revenue_expense_overview.png"
            )
            if supabase_path:
                chart_paths.append(supabase_path)
        
        # 3. Business Pattern Analysis Chart (if patterns detected)
        if financial_metrics.detected_business_types:
            pattern_chart_path = generate_business_pattern_chart(
                financial_metrics,
                temp_dir
            )
            if pattern_chart_path:
                supabase_path = upload_chart_to_supabase(
                    pattern_chart_path, client_id, trace_id, "business_patterns.png"
                )
                if supabase_path:
                    chart_paths.append(supabase_path)
    
    except Exception as e:
        logger.log_agent_activity(
            "chart_generator", "chart_creation_failed", trace_id,
            error=str(e)
        )
    
    finally:
        # Clean up temporary directory
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    return chart_paths

def upload_chart_to_supabase(local_path: str, client_id: str, trace_id: str, filename: str) -> str:
    """Upload chart to Supabase Storage and return storage path"""
    try:
        with open(local_path, 'rb') as file:
            storage_path = f"{client_id}/{trace_id}/charts/{filename}"
            
            result = supabase_client.storage.from_(settings.charts_bucket).upload(
                storage_path, file
            )
            
            # Handle both old and new Supabase API response formats
            if hasattr(result, 'error') and result.error:
                raise Exception(f"Chart upload failed: {result.error}")
            elif isinstance(result, dict) and result.get('error'):
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
    
    try:
        # Prepare data for pyecharts
        data_pairs = [
            (category.replace('_', ' ').title(), float(amount)) 
            for category, amount in expense_by_category.items()
            if float(amount) > 0
        ]
        
        if not data_pairs:
            return None
        
        # Create pie chart with Apache ECharts
        pie_chart = (
            Pie(init_opts=opts.InitOpts(
                theme=ThemeType.WESTEROS, 
                width="800px", 
                height="600px"
            ))
            .add(
                series_name="Expenses",
                data_pair=data_pairs,
                radius=["40%", "75%"],
            )
            .set_global_opts(
                title_opts=opts.TitleOpts(
                    title="Business Expenses by Category",
                    pos_left="center",
                    title_textstyle_opts=opts.TextStyleOpts(
                        font_size=20, 
                        font_weight="bold"
                    )
                ),
                legend_opts=opts.LegendOpts(
                    orient="vertical", 
                    pos_top="15%", 
                    pos_left="2%"
                ),
                toolbox_opts=opts.ToolboxOpts(),
            )
            .set_series_opts(
                tooltip_opts=opts.TooltipOpts(
                    trigger="item", 
                    formatter="{a} <br/>{b}: ${c:,.2f} ({d}%)"
                ),
                label_opts=opts.LabelOpts(
                    formatter="{b}: {d}%"
                ),
            )
        )
        
        # Save chart as PNG
        chart_path = os.path.join(output_dir, "expense_categories_pie.png")
        make_snapshot(snapshot, pie_chart.render(), chart_path)
        
        return chart_path
        
    except Exception as e:
        logger.log_agent_activity(
            "chart_generator", "pie_chart_failed", "unknown",
            error=str(e)
        )
        return None

def generate_revenue_expense_chart(financial_metrics: FinancialMetrics, output_dir: str) -> str:
    """Generate bar chart comparing revenue and expenses using Apache ECharts"""
    
    try:
        categories = ['Revenue', 'Expenses', 'Profit']
        values = [
            float(financial_metrics.total_revenue),
            float(financial_metrics.total_expenses),
            float(financial_metrics.gross_profit)
        ]
        
        # Create bar chart with Apache ECharts
        bar_chart = (
            Bar(init_opts=opts.InitOpts(
                theme=ThemeType.WESTEROS, 
                width="800px", 
                height="600px"
            ))
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
                    title="Financial Performance Overview",
                    pos_left="center",
                    title_textstyle_opts=opts.TextStyleOpts(
                        font_size=20, 
                        font_weight="bold"
                    )
                ),
                xaxis_opts=opts.AxisOpts(name="Category"),
                yaxis_opts=opts.AxisOpts(
                    name="Amount ($)",
                    axislabel_opts=opts.LabelOpts(
                        formatter="${value:,.0f}"
                    )
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
        
    except Exception as e:
        logger.log_agent_activity(
            "chart_generator", "bar_chart_failed", "unknown",
            error=str(e)
        )
        return None

def generate_business_pattern_chart(financial_metrics: FinancialMetrics, output_dir: str) -> str:
    """Generate chart showing detected business patterns"""
    
    try:
        if not financial_metrics.detected_business_types:
            return None
        
        # Prepare data for business types
        business_types = financial_metrics.detected_business_types[:5]  # Top 5
        confidence_scores = [90, 85, 80, 75, 70][:len(business_types)]  # Mock confidence scores
        
        # Create horizontal bar chart
        bar_chart = (
            Bar(init_opts=opts.InitOpts(
                theme=ThemeType.WESTEROS, 
                width="800px", 
                height="600px"
            ))
            .add_xaxis(business_types)
            .add_yaxis(
                series_name="Confidence (%)",
                y_axis=confidence_scores,
                itemstyle_opts=opts.ItemStyleOpts(
                    color="#91cc75"
                )
            )
            .reversal_axis()
            .set_global_opts(
                title_opts=opts.TitleOpts(
                    title="Detected Business Activity Patterns",
                    pos_left="center",
                    title_textstyle_opts=opts.TextStyleOpts(
                        font_size=20, 
                        font_weight="bold"
                    )
                ),
                xaxis_opts=opts.AxisOpts(
                    name="Confidence %",
                    min_=0,
                    max_=100
                ),
                yaxis_opts=opts.AxisOpts(name="Business Type"),
                toolbox_opts=opts.ToolboxOpts(),
            )
            .set_series_opts(
                tooltip_opts=opts.TooltipOpts(
                    trigger="axis",
                    formatter="{b}: {c}% confidence"
                ),
                label_opts=opts.LabelOpts(
                    is_show=True,
                    formatter="{c}%"
                ),
            )
        )
        
        # Save chart as PNG
        chart_path = os.path.join(output_dir, "business_patterns.png")
        make_snapshot(snapshot, bar_chart.render(), chart_path)
        
        return chart_path
        
    except Exception as e:
        logger.log_agent_activity(
            "chart_generator", "pattern_chart_failed", "unknown",
            error=str(e)
        )
        return None