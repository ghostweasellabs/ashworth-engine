"""Inter-agent communication utilities."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from src.workflows.state_schemas import WorkflowState


class AgentCommunicator:
    """Handles communication between agents through workflow state."""
    
    @staticmethod
    def send_message(
        state: WorkflowState, 
        from_agent: str, 
        target_agent: str, 
        message: str, 
        data: Optional[Dict[str, Any]] = None
    ) -> WorkflowState:
        """Send a message to another agent through the workflow state.
        
        Args:
            state: Current workflow state
            from_agent: Sending agent identifier
            target_agent: Target agent identifier
            message: Message content
            data: Optional data payload
            
        Returns:
            Updated workflow state
        """
        if "messages" not in state:
            state["messages"] = []
        
        message_obj = {
            "from": from_agent,
            "to": target_agent,
            "message": message,
            "data": data or {},
            "timestamp": datetime.utcnow().isoformat(),
        }
        
        state["messages"].append(message_obj)
        return state
    
    @staticmethod
    def get_messages_for_agent(state: WorkflowState, agent_id: str) -> List[Dict[str, Any]]:
        """Get messages addressed to a specific agent.
        
        Args:
            state: Current workflow state
            agent_id: Agent ID to get messages for
            
        Returns:
            List of messages
        """
        messages = state.get("messages", [])
        return [msg for msg in messages if msg.get("to") == agent_id]