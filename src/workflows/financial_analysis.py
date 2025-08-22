from langgraph.graph import StateGraph, START, END
from langgraph.func import entrypoint, task
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from src.workflows.state_schemas import OverallState, InputState, OutputState
from typing import Dict, Any
import uuid
from datetime import datetime

@task
def data_fetcher_task(state: OverallState) -> Dict[str, Any]:
    """Extract and normalize financial data from input files"""
    try:
        # Import here to avoid circular imports
        from src.agents.data_fetcher import data_fetcher_agent
        return data_fetcher_agent(state)
    except Exception as e:
        return {
            "raw_extracted_data": [],
            "error_messages": [f"Data fetcher error: {str(e)}"],
            "workflow_phase": "data_extraction_failed"
        }

@task
def data_cleaner_task(state: OverallState) -> Dict[str, Any]:
    """Clean, standardize, and format extracted data for LLM processing"""
    try:
        # Import here to avoid circular imports
        from src.agents.data_cleaner import data_cleaner_agent
        return data_cleaner_agent(state)
    except Exception as e:
        return {
            "transactions": [],
            "error_messages": [f"Data cleaner error: {str(e)}"],
            "workflow_phase": "data_cleaning_failed"
        }

@task  
def data_processor_task(state: OverallState) -> Dict[str, Any]:
    """Process and analyze financial data"""
    try:
        # Import here to avoid circular imports
        from src.agents.data_processor import data_processor_agent
        return data_processor_agent(state)
    except Exception as e:
        return {
            "error_messages": [f"Data processor error: {str(e)}"],
            "workflow_phase": "data_processing_failed"
        }

@task
def tax_categorizer_task(state: OverallState) -> Dict[str, Any]:
    """Categorize transactions for tax compliance"""
    try:
        # Import here to avoid circular imports
        from src.agents.tax_categorizer import tax_categorizer_agent
        return tax_categorizer_agent(state)
    except Exception as e:
        return {
            "error_messages": [f"Tax categorizer error: {str(e)}"],
            "workflow_phase": "tax_categorization_failed"
        }

@task
def report_generator_task(state: OverallState) -> Dict[str, Any]:
    """Generate consulting-grade narrative report"""
    try:
        # Import here to avoid circular imports
        from src.agents.report_generator import report_generator_agent
        return report_generator_agent(state)
    except Exception as e:
        return {
            "error_messages": [f"Report generator error: {str(e)}"],
            "workflow_phase": "report_generation_failed"
        }

# Build StateGraph with proper configuration
graph_builder = StateGraph(
    OverallState,
    input_schema=InputState,
    output_schema=OutputState
)

# Add nodes with error handling
graph_builder.add_node("data_fetcher", data_fetcher_task)
graph_builder.add_node("data_cleaner", data_cleaner_task)
graph_builder.add_node("data_processor", data_processor_task)
graph_builder.add_node("tax_categorizer", tax_categorizer_task)
graph_builder.add_node("report_generator", report_generator_task)

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
graph_builder.add_edge(START, "data_fetcher")
graph_builder.add_conditional_edges(
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

graph_builder.add_conditional_edges(
    "data_cleaner",
    route_from_cleaner,
    {
        "data_processor": "data_processor",
        END: END
    }
)
graph_builder.add_edge("data_processor", "tax_categorizer")
graph_builder.add_edge("tax_categorizer", "report_generator")
graph_builder.add_edge("report_generator", END)

# Compile with checkpointer and store
checkpointer = InMemorySaver()
store = InMemoryStore()

app = graph_builder.compile(
    checkpointer=checkpointer,
    store=store
)