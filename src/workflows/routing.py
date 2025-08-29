"""Conditional routing logic for LangGraph workflows."""

import logging
from typing import Dict, Any, List
from src.workflows.state_schemas import WorkflowState, AgentStatus

logger = logging.getLogger(__name__)


class WorkflowRouter:
    """Handles conditional routing decisions in workflows."""
    
    @staticmethod
    def should_continue_to_next_agent(
        state: WorkflowState, 
        current_agent: str,
        next_agent: str
    ) -> bool:
        """Determine if workflow should continue to next agent.
        
        Args:
            state: Current workflow state
            current_agent: Current agent identifier
            next_agent: Next agent identifier
            
        Returns:
            True if should continue, False otherwise
        """
        # Check if current agent completed successfully
        current_status = state.get(current_agent, {}).get("status")
        if current_status != AgentStatus.COMPLETED:
            logger.warning(f"Agent {current_agent} did not complete successfully: {current_status}")
            return False
        
        # Check if there are critical errors
        errors = state.get("errors", [])
        critical_errors = [e for e in errors if "critical" in e.lower() or "fatal" in e.lower()]
        if critical_errors:
            logger.error(f"Critical errors prevent continuation: {critical_errors}")
            return False
        
        # Check data quality requirements for specific transitions
        if current_agent == "data_fetcher" and next_agent == "data_processor":
            return WorkflowRouter._check_data_fetcher_output(state)
        elif current_agent == "data_processor" and next_agent == "categorizer":
            return WorkflowRouter._check_data_processor_output(state)
        elif current_agent == "categorizer" and next_agent == "report_generator":
            return WorkflowRouter._check_categorizer_output(state)
        
        return True
    
    @staticmethod
    def _check_data_fetcher_output(state: WorkflowState) -> bool:
        """Check if data fetcher output is sufficient for processing."""
        file_processing = state.get("file_processing", {})
        
        # Check if we have processed data
        processed_data = file_processing.get("processed_data", [])
        if not processed_data:
            logger.error("No processed data from data fetcher")
            return False
        
        # Check validation errors
        validation_errors = file_processing.get("validation_errors", [])
        if len(validation_errors) > 10:  # Too many validation errors
            logger.error(f"Too many validation errors: {len(validation_errors)}")
            return False
        
        return True
    
    @staticmethod
    def _check_data_processor_output(state: WorkflowState) -> bool:
        """Check if data processor output is sufficient for categorization."""
        analysis = state.get("analysis", {})
        
        # Check if we have transactions
        transactions = analysis.get("transactions", [])
        if not transactions:
            logger.error("No transactions from data processor")
            return False
        
        # Check data quality score
        quality_score = state.get("quality_score", 0.0)
        if quality_score < 0.3:
            logger.error(f"Data quality too low for categorization: {quality_score}")
            return False
        
        return True
    
    @staticmethod
    def _check_categorizer_output(state: WorkflowState) -> bool:
        """Check if categorizer output is sufficient for report generation."""
        analysis = state.get("analysis", {})
        
        # Check if we have categories
        categories = analysis.get("categories", {})
        if not categories:
            logger.warning("No categories from categorizer, but continuing to report")
            # Don't block report generation - it can handle missing categories
        
        # Check for compliance issues that would block reporting
        compliance_issues = analysis.get("compliance_issues", [])
        blocking_issues = [issue for issue in compliance_issues if "blocking" in issue.lower()]
        if blocking_issues:
            logger.error(f"Blocking compliance issues: {blocking_issues}")
            return False
        
        return True
    
    @staticmethod
    def determine_retry_strategy(state: WorkflowState) -> str:
        """Determine which agent to retry based on error analysis.
        
        Args:
            state: Current workflow state
            
        Returns:
            Agent to retry or "end" if no retry should be attempted
        """
        errors = state.get("errors", [])
        
        # Analyze error patterns to determine retry strategy
        for error in errors:
            error_lower = error.lower()
            
            # File processing errors - retry data fetcher
            if any(keyword in error_lower for keyword in ["file", "upload", "parse", "format"]):
                retry_count = state.get("data_fetcher", {}).get("retry_count", 0)
                if retry_count < 3:
                    logger.info("Retrying data fetcher due to file processing error")
                    return "retry_data_fetcher"
            
            # Data processing errors - retry data processor
            elif any(keyword in error_lower for keyword in ["validation", "calculation", "decimal"]):
                retry_count = state.get("data_processor", {}).get("retry_count", 0)
                if retry_count < 2:
                    logger.info("Retrying data processor due to validation error")
                    return "retry_data_processor"
            
            # Categorization errors - retry categorizer
            elif any(keyword in error_lower for keyword in ["category", "tax", "compliance"]):
                retry_count = state.get("categorizer", {}).get("retry_count", 0)
                if retry_count < 2:
                    logger.info("Retrying categorizer due to categorization error")
                    return "retry_categorizer"
            
            # Report generation errors - retry report generator
            elif any(keyword in error_lower for keyword in ["report", "storage", "template"]):
                retry_count = state.get("report_generator", {}).get("retry_count", 0)
                if retry_count < 2:
                    logger.info("Retrying report generator due to report error")
                    return "retry_report_generator"
        
        # No retry strategy found
        logger.info("No retry strategy determined, ending workflow")
        return "end"
    
    @staticmethod
    def should_interrupt_for_review(state: WorkflowState, checkpoint: str) -> bool:
        """Determine if workflow should be interrupted for human review.
        
        Args:
            state: Current workflow state
            checkpoint: Checkpoint name
            
        Returns:
            True if should interrupt, False otherwise
        """
        # Always interrupt before report generation for high-value workflows
        if checkpoint == "report_generator":
            config = state.get("config", {})
            if config.get("require_human_review", False):
                return True
            
            # Interrupt if there are warnings or quality issues
            warnings = state.get("warnings", [])
            if warnings:
                logger.info(f"Interrupting for review due to warnings: {warnings}")
                return True
        
        # Interrupt after quality check if quality is low
        if checkpoint == "quality_check":
            quality_score = state.get("quality_score", 1.0)
            if quality_score < 0.7:
                logger.info(f"Interrupting for review due to low quality: {quality_score}")
                return True
        
        return False
    
    @staticmethod
    def calculate_confidence_score(state: WorkflowState) -> float:
        """Calculate overall confidence score for the workflow results.
        
        Args:
            state: Current workflow state
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        confidence_factors = []
        
        # Data quality factor
        quality_score = state.get("quality_score", 1.0)
        confidence_factors.append(quality_score)
        
        # Agent completion factor
        completed_agents = 0
        total_agents = 0
        for agent_key in ["data_fetcher", "data_processor", "categorizer", "report_generator"]:
            total_agents += 1
            agent_state = state.get(agent_key, {})
            if agent_state.get("status") == AgentStatus.COMPLETED:
                completed_agents += 1
        
        completion_factor = completed_agents / total_agents if total_agents > 0 else 0.0
        confidence_factors.append(completion_factor)
        
        # Error factor
        errors = state.get("errors", [])
        warnings = state.get("warnings", [])
        error_factor = max(0.0, 1.0 - (len(errors) * 0.3) - (len(warnings) * 0.15))
        confidence_factors.append(error_factor)
        
        # Transaction coverage factor
        analysis = state.get("analysis", {})
        transactions = analysis.get("transactions", [])
        categories = analysis.get("categories", {})
        
        if transactions:
            categorized_count = len([t for t in transactions if t.get("category")])
            coverage_factor = categorized_count / len(transactions)
            confidence_factors.append(coverage_factor)
        
        # Calculate weighted average
        if confidence_factors:
            confidence_score = sum(confidence_factors) / len(confidence_factors)
        else:
            confidence_score = 0.0
        
        return min(1.0, max(0.0, confidence_score))


# Convenience functions for use in workflow definitions
def should_generate_report(state: WorkflowState) -> str:
    """Routing function for quality check to report generation."""
    
    # Check if any critical agents failed
    critical_agents = ["data_fetcher", "data_processor", "categorizer"]
    for agent in critical_agents:
        if state.get(agent, {}).get("status") == AgentStatus.FAILED:
            return "handle_error"
    
    # Check if we have sufficient data quality
    quality_score = state.get("quality_score", 1.0)
    if quality_score < 0.3:
        return "handle_error"
    
    # Check if we have transactions to report on
    transactions = state.get("analysis", {}).get("transactions", [])
    if not transactions:
        return "handle_error"
    
    return "generate_report"


def check_report_generation(state: WorkflowState) -> str:
    """Routing function for report generator completion."""
    
    report_status = state.get("report_generator", {}).get("status")
    if report_status == AgentStatus.FAILED:
        return "error"
    
    # Check if report was actually generated
    if not state.get("output_reports"):
        return "error"
    
    return "success"


def should_retry(state: WorkflowState) -> str:
    """Routing function for error handler retry logic."""
    return WorkflowRouter.determine_retry_strategy(state)