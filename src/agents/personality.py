"""Agent personality system for behavior and prompt management."""

import logging
from typing import Any, Dict, List


class AgentPersonality:
    """Agent personality configuration with prompts and behavior patterns."""
    
    def __init__(
        self,
        name: str,
        title: str,
        background: str,
        personality_traits: List[str],
        communication_style: str,
        expertise_areas: List[str],
        system_prompt: str,
        task_prompt_template: str,
        error_handling_style: str = "professional"
    ):
        self.name = name
        self.title = title
        self.background = background
        self.personality_traits = personality_traits
        self.communication_style = communication_style
        self.expertise_areas = expertise_areas
        self.system_prompt = system_prompt
        self.task_prompt_template = task_prompt_template
        self.error_handling_style = error_handling_style
    
    def format_task_prompt(self, task_context: Dict[str, Any]) -> str:
        """Format the task prompt with context variables."""
        try:
            return self.task_prompt_template.format(**task_context)
        except KeyError as e:
            logging.warning(f"Missing context variable for {self.name}: {e}")
            return self.task_prompt_template
    
    def get_error_response(self, error: Exception, context: Dict[str, Any]) -> str:
        """Generate personality-appropriate error response."""
        error_responses = {
            "analytical": f"Analysis indicates a processing anomaly: {str(error)}. Investigating root cause and implementing corrective measures.",
            "methodical": f"Data integrity breach detected: {str(error)}. Initiating validation protocols and error recovery procedures.",
            "strategic": f"Compliance risk identified: {str(error)}. Recommending conservative approach and professional consultation.",
            "executive": f"Processing challenge encountered: {str(error)}. Implementing alternative approach to maintain deliverable quality.",
        }
        
        return error_responses.get(
            self.error_handling_style, 
            f"Processing error encountered: {str(error)}. Implementing recovery procedures."
        )