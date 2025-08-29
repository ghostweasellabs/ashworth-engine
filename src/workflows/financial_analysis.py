"""Financial analysis workflow using LangGraph StateGraph."""

import logging
from typing import Any, Dict, List, Optional
from datetime import datetime

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode

from src.workflows.state_schemas import (
    WorkflowState, 
    WorkflowStatus, 
    AgentStatus,
    create_initial_workflow_state,
    update_agent_state
)
from src.workflows.routing import (
    should_generate_report,
    check_report_generation,
    should_retry,
    WorkflowRouter
)
from src.agents.data_fetcher import DataFetcherAgent
from src.agents.data_processor import DataProcessorAgent  
from src.agents.categorizer import CategorizerAgent
from src.agents.report_generator import ReportGeneratorAgent

logger = logging.getLogger(__name__)


class FinancialAnalysisWorkflow:
    """LangGraph workflow for financial analysis with sequential agent execution."""
    
    def __init__(self, checkpointer: Optional[MemorySaver] = None):
        """Initialize the workflow.
        
        Args:
            checkpointer: Optional checkpointer for state persistence
        """
        self.checkpointer = checkpointer or MemorySaver()
        self.graph = self._build_graph()
        
        # Initialize agents
        self.data_fetcher = DataFetcherAgent()
        self.data_processor = DataProcessorAgent()
        self.categorizer = CategorizerAgent()
        self.report_generator = ReportGeneratorAgent()
    
    def _build_graph(self) -> StateGraph:
        """Build the StateGraph workflow with sequential execution."""
        
        # Create the graph
        workflow = StateGraph(WorkflowState)
        
        # Add agent nodes
        workflow.add_node("data_fetcher", self._data_fetcher_node)
        workflow.add_node("data_processor", self._data_processor_node)
        workflow.add_node("categorizer", self._categorizer_node)
        workflow.add_node("report_generator", self._report_generator_node)
        workflow.add_node("error_handler", self._error_handler_node)
        workflow.add_node("quality_check", self._quality_check_node)
        
        # Define sequential workflow edges
        workflow.add_edge(START, "data_fetcher")
        workflow.add_edge("data_fetcher", "data_processor")
        workflow.add_edge("data_processor", "categorizer")
        workflow.add_edge("categorizer", "quality_check")
        
        # Add conditional routing from quality check
        workflow.add_conditional_edges(
            "quality_check",
            should_generate_report,
            {
                "generate_report": "report_generator",
                "handle_error": "error_handler",
                "end": END
            }
        )
        
        # Report generator can go to end or error handler
        workflow.add_conditional_edges(
            "report_generator",
            check_report_generation,
            {
                "success": END,
                "error": "error_handler"
            }
        )
        
        # Error handler can retry or end
        workflow.add_conditional_edges(
            "error_handler",
            should_retry,
            {
                "retry_data_fetcher": "data_fetcher",
                "retry_data_processor": "data_processor", 
                "retry_categorizer": "categorizer",
                "retry_report_generator": "report_generator",
                "end": END
            }
        )
        
        return workflow
    
    async def _data_fetcher_node(self, state: WorkflowState) -> WorkflowState:
        """Execute data fetcher agent."""
        logger.info(f"Starting data fetcher for workflow {state['workflow_id']}")
        
        try:
            # Update agent status to in progress
            state = update_agent_state(state, "data_fetcher", AgentStatus.IN_PROGRESS)
            
            # Execute data fetcher
            result = await self.data_fetcher.process(
                input_files=state.get("input_files", []),
                config=state.get("config", {})
            )
            
            # Update state with results
            state["file_processing"] = result.get("file_processing", {})
            state = update_agent_state(
                state, 
                "data_fetcher", 
                AgentStatus.COMPLETED,
                output_data=result
            )
            
            logger.info(f"Data fetcher completed for workflow {state['workflow_id']}")
            
        except Exception as e:
            logger.error(f"Data fetcher failed: {str(e)}")
            state = update_agent_state(
                state,
                "data_fetcher", 
                AgentStatus.FAILED,
                error_message=str(e)
            )
            state["errors"].append(f"Data fetcher failed: {str(e)}")
        
        return state
    
    async def _data_processor_node(self, state: WorkflowState) -> WorkflowState:
        """Execute data processor agent."""
        logger.info(f"Starting data processor for workflow {state['workflow_id']}")
        
        try:
            # Check if data fetcher completed successfully
            if state.get("data_fetcher", {}).get("status") != AgentStatus.COMPLETED:
                raise ValueError("Data fetcher must complete successfully before data processor")
            
            # Update agent status
            state = update_agent_state(state, "data_processor", AgentStatus.IN_PROGRESS)
            
            # Execute data processor
            result = await self.data_processor.process(
                raw_data=state.get("file_processing", {}).get("processed_data", []),
                config=state.get("config", {})
            )
            
            # Update state with results
            state["analysis"] = result.get("analysis", {})
            state = update_agent_state(
                state,
                "data_processor",
                AgentStatus.COMPLETED,
                output_data=result
            )
            
            logger.info(f"Data processor completed for workflow {state['workflow_id']}")
            
        except Exception as e:
            logger.error(f"Data processor failed: {str(e)}")
            state = update_agent_state(
                state,
                "data_processor",
                AgentStatus.FAILED, 
                error_message=str(e)
            )
            state["errors"].append(f"Data processor failed: {str(e)}")
        
        return state
    
    async def _categorizer_node(self, state: WorkflowState) -> WorkflowState:
        """Execute categorizer agent."""
        logger.info(f"Starting categorizer for workflow {state['workflow_id']}")
        
        try:
            # Check if data processor completed successfully
            if state.get("data_processor", {}).get("status") != AgentStatus.COMPLETED:
                raise ValueError("Data processor must complete successfully before categorizer")
            
            # Update agent status
            state = update_agent_state(state, "categorizer", AgentStatus.IN_PROGRESS)
            
            # Execute categorizer
            result = await self.categorizer.process(
                transactions=state.get("analysis", {}).get("transactions", []),
                config=state.get("config", {})
            )
            
            # Update analysis state with categorization results
            analysis_state = state.get("analysis", {})
            analysis_state.update({
                "categories": result.get("categories", {}),
                "tax_implications": result.get("tax_implications", {}),
                "compliance_issues": result.get("compliance_issues", [])
            })
            state["analysis"] = analysis_state
            
            state = update_agent_state(
                state,
                "categorizer",
                AgentStatus.COMPLETED,
                output_data=result
            )
            
            logger.info(f"Categorizer completed for workflow {state['workflow_id']}")
            
        except Exception as e:
            logger.error(f"Categorizer failed: {str(e)}")
            state = update_agent_state(
                state,
                "categorizer",
                AgentStatus.FAILED,
                error_message=str(e)
            )
            state["errors"].append(f"Categorizer failed: {str(e)}")
        
        return state
    
    async def _report_generator_node(self, state: WorkflowState) -> WorkflowState:
        """Execute report generator agent."""
        logger.info(f"Starting report generator for workflow {state['workflow_id']}")
        
        try:
            # Update agent status
            state = update_agent_state(state, "report_generator", AgentStatus.IN_PROGRESS)
            
            # Execute report generator
            result = await self.report_generator.process(
                analysis_data=state.get("analysis", {}),
                config=state.get("config", {})
            )
            
            # Update state with report results
            state["report"] = result.get("report", {})
            state["output_reports"] = result.get("output_reports", [])
            
            state = update_agent_state(
                state,
                "report_generator", 
                AgentStatus.COMPLETED,
                output_data=result
            )
            
            logger.info(f"Report generator completed for workflow {state['workflow_id']}")
            
        except Exception as e:
            logger.error(f"Report generator failed: {str(e)}")
            state = update_agent_state(
                state,
                "report_generator",
                AgentStatus.FAILED,
                error_message=str(e)
            )
            state["errors"].append(f"Report generator failed: {str(e)}")
        
        return state
    
    async def _quality_check_node(self, state: WorkflowState) -> WorkflowState:
        """Perform quality checks before report generation."""
        logger.info(f"Performing quality check for workflow {state['workflow_id']}")
        
        try:
            quality_issues = []
            
            # Check if categorizer completed successfully
            if state.get("categorizer", {}).get("status") != AgentStatus.COMPLETED:
                quality_issues.append("Categorizer did not complete successfully")
            
            # Check if we have transactions to report on
            transactions = state.get("analysis", {}).get("transactions", [])
            if not transactions:
                quality_issues.append("No transactions found for analysis")
            
            # Check data quality score
            quality_score = state.get("quality_score", 0.0)
            if quality_score < 0.5:
                quality_issues.append(f"Data quality score too low: {quality_score}")
            
            # Update state with quality issues
            if quality_issues:
                state["warnings"].extend(quality_issues)
                state["quality_score"] = max(0.0, state.get("quality_score", 1.0) - 0.1)
            
            logger.info(f"Quality check completed with {len(quality_issues)} issues")
            
        except Exception as e:
            logger.error(f"Quality check failed: {str(e)}")
            state["errors"].append(f"Quality check failed: {str(e)}")
        
        return state
    
    async def _error_handler_node(self, state: WorkflowState) -> WorkflowState:
        """Handle errors and determine retry strategy."""
        logger.info(f"Handling errors for workflow {state['workflow_id']}")
        
        try:
            errors = state.get("errors", [])
            
            # Log all errors
            for error in errors:
                logger.error(f"Workflow error: {error}")
            
            # Update workflow status
            state["status"] = WorkflowStatus.FAILED
            state["completed_at"] = datetime.utcnow()
            
            # Calculate total duration
            if state.get("started_at"):
                duration = (state["completed_at"] - state["started_at"]).total_seconds() * 1000
                state["duration_ms"] = int(duration)
            
            logger.info(f"Error handling completed for workflow {state['workflow_id']}")
            
        except Exception as e:
            logger.error(f"Error handler failed: {str(e)}")
            state["errors"].append(f"Error handler failed: {str(e)}")
        
        return state
    

    
    def compile(self) -> StateGraph:
        """Compile the workflow graph with checkpointing."""
        
        return self.graph.compile(
            checkpointer=self.checkpointer,
            interrupt_before=["report_generator"],  # Allow human review before final report
            interrupt_after=["quality_check"]      # Allow review after quality check
        )
    
    async def execute(
        self, 
        workflow_id: str,
        input_files: List[str],
        config: Optional[Dict[str, Any]] = None
    ) -> WorkflowState:
        """Execute the complete workflow.
        
        Args:
            workflow_id: Unique workflow identifier
            input_files: List of input file paths
            config: Optional workflow configuration
            
        Returns:
            Final workflow state
        """
        logger.info(f"Starting workflow execution: {workflow_id}")
        
        # Create initial state
        initial_state = create_initial_workflow_state(
            workflow_id=workflow_id,
            input_files=input_files,
            config=config or {}
        )
        initial_state["status"] = WorkflowStatus.RUNNING
        
        # Compile and execute workflow
        app = self.compile()
        
        try:
            # Execute the workflow
            final_state = await app.ainvoke(
                initial_state,
                config={"configurable": {"thread_id": workflow_id}}
            )
            
            # Update final status
            if final_state.get("status") != WorkflowStatus.FAILED:
                final_state["status"] = WorkflowStatus.COMPLETED
                final_state["completed_at"] = datetime.utcnow()
                
                # Calculate total duration
                if final_state.get("started_at"):
                    duration = (final_state["completed_at"] - final_state["started_at"]).total_seconds() * 1000
                    final_state["duration_ms"] = int(duration)
            
            logger.info(f"Workflow execution completed: {workflow_id}")
            return final_state
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {str(e)}")
            
            # Update state with failure
            initial_state["status"] = WorkflowStatus.FAILED
            initial_state["completed_at"] = datetime.utcnow()
            initial_state["errors"].append(f"Workflow execution failed: {str(e)}")
            
            return initial_state


# Factory function for creating workflow instances
def create_financial_analysis_workflow(
    checkpointer: Optional[MemorySaver] = None
) -> FinancialAnalysisWorkflow:
    """Create a new financial analysis workflow instance.
    
    Args:
        checkpointer: Optional checkpointer for state persistence
        
    Returns:
        Configured workflow instance
    """
    return FinancialAnalysisWorkflow(checkpointer=checkpointer)