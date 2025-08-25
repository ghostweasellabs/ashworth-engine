from langgraph.graph import StateGraph, START, END
from langchain_core.runnables import RunnableConfig
from src.workflows.state_schemas import OverallState, InputState, OutputState
from src.utils.checkpointer import get_shared_checkpointer
from src.utils.memory_store import get_shared_memory_store
from src.config.settings import settings
from typing import Dict, Any, Optional
import uuid
from datetime import datetime

def data_fetcher_node(state: OverallState, config: Optional[RunnableConfig] = None, *, store=None) -> Dict[str, Any]:
    """Extract and normalize financial data from input files"""
    try:
        # Import here to avoid circular imports
        from src.agents.data_fetcher import data_fetcher_agent
        return data_fetcher_agent(state, config, store=store)
    except Exception as e:
        return {
            "raw_extracted_data": [],
            "error_messages": [f"Data fetcher error: {str(e)}"],
            "workflow_phase": "data_extraction_failed"
        }

def data_cleaner_node(state: OverallState, config: Optional[RunnableConfig] = None, *, store=None) -> Dict[str, Any]:
    """Clean, standardize, and format extracted data for LLM processing"""
    try:
        # Import here to avoid circular imports
        from src.agents.data_cleaner import data_cleaner_agent
        return data_cleaner_agent(state, config, store=store)
    except Exception as e:
        return {
            "transactions": [],
            "error_messages": [f"Data cleaner error: {str(e)}"],
            "workflow_phase": "data_cleaning_failed"
        }

def data_processor_node(state: OverallState, config: Optional[RunnableConfig] = None, *, store=None) -> Dict[str, Any]:
    """Process and analyze financial data"""
    try:
        # Import here to avoid circular imports
        from src.agents.data_processor import data_processor_agent
        return data_processor_agent(state, config, store=store)
    except Exception as e:
        return {
            "error_messages": [f"Data processor error: {str(e)}"],
            "workflow_phase": "data_processing_failed"
        }

def tax_categorizer_node(state: OverallState, config: Optional[RunnableConfig] = None, *, store=None) -> Dict[str, Any]:
    """Categorize transactions for tax compliance"""
    try:
        # Import here to avoid circular imports
        from src.agents.tax_categorizer import tax_categorizer_agent
        return tax_categorizer_agent(state, config, store=store)
    except Exception as e:
        return {
            "error_messages": [f"Tax categorizer error: {str(e)}"],
            "workflow_phase": "tax_categorization_failed"
        }

def report_generator_node(state: OverallState, config: Optional[RunnableConfig] = None, *, store=None) -> Dict[str, Any]:
    """Generate consulting-grade narrative report"""
    try:
        # Import here to avoid circular imports
        from src.agents.report_generator import report_generator_agent
        return report_generator_agent(state, config, store=store)
    except Exception as e:
        return {
            "error_messages": [f"Report generator error: {str(e)}"],
            "workflow_phase": "report_generation_failed"
        }

def chart_generator_node(state: OverallState, config: Optional[RunnableConfig] = None, *, store=None) -> Dict[str, Any]:
    """Generate professional charts and visualizations"""
    try:
        # Import here to avoid circular imports
        from src.agents.chart_generator import chart_generator_agent
        return chart_generator_agent(state, config, store=store)
    except Exception as e:
        return {
            "error_messages": [f"Chart generator error: {str(e)}"],
            "workflow_phase": "chart_generation_failed"
        }

# Build StateGraph with proper configuration
builder = StateGraph(OverallState)

# Add nodes with error handling
builder.add_node("data_fetcher", data_fetcher_node)
builder.add_node("data_cleaner", data_cleaner_node)
builder.add_node("data_processor", data_processor_node)
builder.add_node("tax_categorizer", tax_categorizer_node)
builder.add_node("chart_generator", chart_generator_node)
builder.add_node("report_generator", report_generator_node)

# Define workflow edges based on analysis_type
def route_workflow(state: OverallState) -> str:
    """Route based on analysis_type parameter"""
    analysis_type = state.get("analysis_type", "financial_analysis")
    
    if analysis_type == "data_collection":
        return END  # Stop after data fetcher
    elif analysis_type == "data_cleaning":
        return "data_cleaner"  # Stop after data cleaning
    elif analysis_type == "data_processing":
        return "data_cleaner"  # Continue to cleaning then processing
    elif analysis_type == "tax_categorization":
        return "data_cleaner"  # Continue full workflow
    else:  # financial_analysis or strategic_planning
        return "data_cleaner"

# Add edges
builder.add_edge(START, "data_fetcher")
builder.add_conditional_edges(
    "data_fetcher",
    route_workflow,
    {
        "data_cleaner": "data_cleaner",
        END: END
    }
)

# Route from data_cleaner based on analysis_type
def route_from_cleaner(state: OverallState) -> str:
    """Route from data_cleaner based on analysis_type"""
    analysis_type = state.get("analysis_type", "financial_analysis")
    
    if analysis_type == "data_cleaning":
        return END  # Stop after cleaning
    else:
        return "data_processor"  # Continue to processing

builder.add_conditional_edges(
    "data_cleaner",
    route_from_cleaner,
    {
        "data_processor": "data_processor",
        END: END
    }
)
builder.add_edge("data_processor", "tax_categorizer")
builder.add_edge("tax_categorizer", "chart_generator")
builder.add_edge("chart_generator", "report_generator")
builder.add_edge("report_generator", END)

# Compile with shared checkpointer and store
def create_financial_analysis_workflow():
    """Create the financial analysis workflow with shared memory and persistence"""
    
    # Get shared checkpointer and memory store
    if settings.enable_checkpointing:
        shared_checkpointer = get_shared_checkpointer()
        checkpointer = shared_checkpointer.get_checkpointer()
    else:
        from langgraph.checkpoint.memory import InMemorySaver
        checkpointer = InMemorySaver()
    
    if settings.enable_shared_memory:
        shared_memory = get_shared_memory_store()
        store = shared_memory.get_store()
    else:
        from langgraph.store.memory import InMemoryStore
        store = InMemoryStore()
    
    return builder.compile(
        checkpointer=checkpointer,
        store=store
    )

# Create the compiled workflow
app = create_financial_analysis_workflow()