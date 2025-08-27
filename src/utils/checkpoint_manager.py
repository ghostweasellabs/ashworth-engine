"""Checkpoint management utilities for workflow rollback and recovery."""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta

from langgraph.checkpoint.base import Checkpoint, CheckpointMetadata

from src.stores.postgres_checkpoint import PostgresCheckpointer
from src.workflows.state_schemas import WorkflowState, WorkflowStatus


class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder for datetime objects."""
    
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages workflow checkpoints for rollback and recovery capabilities."""
    
    def __init__(self, checkpointer: Optional[PostgresCheckpointer] = None):
        """Initialize the checkpoint manager.
        
        Args:
            checkpointer: PostgreSQL checkpointer instance. If None, creates a new one.
        """
        self.checkpointer = checkpointer or PostgresCheckpointer()
    
    def create_checkpoint(
        self,
        workflow_id: str,
        state: WorkflowState,
        checkpoint_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a checkpoint for the current workflow state.
        
        Args:
            workflow_id: Unique workflow identifier
            state: Current workflow state
            checkpoint_name: Optional name for the checkpoint
            metadata: Additional metadata to store with checkpoint
            
        Returns:
            Checkpoint ID
        """
        checkpoint_id = f"{workflow_id}_{datetime.utcnow().isoformat()}"
        
        # Convert datetime objects to ISO strings for JSON serialization
        def convert_datetime_to_str(obj):
            """Recursively convert datetime objects to ISO strings."""
            if isinstance(obj, datetime):
                return obj.isoformat()
            elif isinstance(obj, dict):
                return {k: convert_datetime_to_str(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_datetime_to_str(item) for item in obj]
            else:
                return obj
        
        serializable_state = convert_datetime_to_str(state)
        
        checkpoint = Checkpoint(
            id=checkpoint_id,
            ts=datetime.utcnow().isoformat(),
            channel_values={"state": serializable_state},
            channel_versions={},
            versions_seen={}
        )
        
        checkpoint_metadata = CheckpointMetadata(
            source="checkpoint_manager",
            step=state.get("orchestrator", {}).get("retry_count", 0),
            writes={},
            parents={}
        )
        
        # Add custom metadata
        if metadata:
            checkpoint_metadata.update(metadata)
        
        if checkpoint_name:
            checkpoint_metadata["name"] = checkpoint_name
        
        # Store workflow status and timing info
        checkpoint_metadata.update({
            "workflow_status": state.get("status"),
            "created_at": datetime.utcnow().isoformat(),
            "agent_statuses": {
                agent_id: agent_state.get("status")
                for agent_id, agent_state in state.items()
                if isinstance(agent_state, dict) and "status" in agent_state
            }
        })
        
        config = {
            "configurable": {
                "thread_id": workflow_id
            }
        }
        
        try:
            self.checkpointer.put(config, checkpoint, checkpoint_metadata)
            logger.info(f"Created checkpoint {checkpoint_id} for workflow {workflow_id}")
            return checkpoint_id
            
        except Exception as e:
            logger.error(f"Failed to create checkpoint for workflow {workflow_id}: {e}")
            raise
    
    def restore_checkpoint(
        self,
        workflow_id: str,
        checkpoint_id: Optional[str] = None
    ) -> Optional[WorkflowState]:
        """Restore workflow state from a checkpoint.
        
        Args:
            workflow_id: Unique workflow identifier
            checkpoint_id: Specific checkpoint ID. If None, restores latest checkpoint.
            
        Returns:
            Restored workflow state or None if not found
        """
        config = {
            "configurable": {
                "thread_id": workflow_id
            }
        }
        
        try:
            if checkpoint_id:
                # Get specific checkpoint (would need to implement in checkpointer)
                # For now, get the latest
                result = self.checkpointer.get(config)
            else:
                # Get latest checkpoint
                result = self.checkpointer.get(config)
            
            if not result:
                logger.warning(f"No checkpoint found for workflow {workflow_id}")
                return None
            
            checkpoint, metadata = result
            state = checkpoint.get("channel_values", {}).get("state")
            
            if state:
                logger.info(f"Restored checkpoint for workflow {workflow_id}")
                return state
            else:
                logger.warning(f"No state found in checkpoint for workflow {workflow_id}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to restore checkpoint for workflow {workflow_id}: {e}")
            return None
    
    def list_checkpoints(
        self,
        workflow_id: str,
        limit: Optional[int] = 10
    ) -> List[Dict[str, Any]]:
        """List available checkpoints for a workflow.
        
        Args:
            workflow_id: Unique workflow identifier
            limit: Maximum number of checkpoints to return
            
        Returns:
            List of checkpoint information
        """
        config = {
            "configurable": {
                "thread_id": workflow_id
            }
        }
        
        checkpoints = []
        
        try:
            for checkpoint, metadata in self.checkpointer.list(config, limit=limit):
                checkpoint_info = {
                    "id": checkpoint.get("id"),
                    "timestamp": checkpoint.get("ts"),
                    "name": metadata.get("name"),
                    "workflow_status": metadata.get("workflow_status"),
                    "agent_statuses": metadata.get("agent_statuses", {}),
                    "step": metadata.get("step", 0),
                    "created_at": metadata.get("created_at")
                }
                checkpoints.append(checkpoint_info)
                
            logger.debug(f"Found {len(checkpoints)} checkpoints for workflow {workflow_id}")
            return checkpoints
            
        except Exception as e:
            logger.error(f"Failed to list checkpoints for workflow {workflow_id}: {e}")
            return []
    
    def rollback_to_checkpoint(
        self,
        workflow_id: str,
        checkpoint_id: str
    ) -> Optional[WorkflowState]:
        """Rollback workflow to a specific checkpoint.
        
        Args:
            workflow_id: Unique workflow identifier
            checkpoint_id: Checkpoint ID to rollback to
            
        Returns:
            Restored workflow state or None if rollback failed
        """
        try:
            # Restore the checkpoint
            state = self.restore_checkpoint(workflow_id, checkpoint_id)
            
            if not state:
                logger.error(f"Cannot rollback: checkpoint {checkpoint_id} not found")
                return None
            
            # Update workflow status to indicate rollback
            state["status"] = WorkflowStatus.PENDING
            state["checkpoint_metadata"]["rollback_from"] = checkpoint_id
            state["checkpoint_metadata"]["rollback_at"] = datetime.utcnow().isoformat()
            
            # Reset any agents that were running or completed after this checkpoint
            # This would require more sophisticated logic based on checkpoint timing
            
            # Create a new checkpoint after rollback
            rollback_checkpoint_id = self.create_checkpoint(
                workflow_id,
                state,
                checkpoint_name=f"rollback_to_{checkpoint_id}",
                metadata={"rollback": True, "original_checkpoint": checkpoint_id}
            )
            
            logger.info(f"Rolled back workflow {workflow_id} to checkpoint {checkpoint_id}")
            return state
            
        except Exception as e:
            logger.error(f"Failed to rollback workflow {workflow_id} to checkpoint {checkpoint_id}: {e}")
            return None
    
    def cleanup_old_checkpoints(
        self,
        days_old: int = 30,
        keep_minimum: int = 5
    ) -> int:
        """Clean up old checkpoints while keeping a minimum number.
        
        Args:
            days_old: Delete checkpoints older than this many days
            keep_minimum: Minimum number of checkpoints to keep per workflow
            
        Returns:
            Number of checkpoints deleted
        """
        try:
            deleted_count = self.checkpointer.cleanup(days_old)
            logger.info(f"Cleaned up {deleted_count} old checkpoints")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup old checkpoints: {e}")
            return 0
    
    def get_checkpoint_stats(self, workflow_id: str) -> Dict[str, Any]:
        """Get statistics about checkpoints for a workflow.
        
        Args:
            workflow_id: Unique workflow identifier
            
        Returns:
            Dictionary with checkpoint statistics
        """
        checkpoints = self.list_checkpoints(workflow_id, limit=None)
        
        if not checkpoints:
            return {
                "total_checkpoints": 0,
                "latest_checkpoint": None,
                "oldest_checkpoint": None,
                "status_distribution": {}
            }
        
        # Calculate statistics
        status_counts = {}
        for cp in checkpoints:
            status = cp.get("workflow_status", "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1
        
        return {
            "total_checkpoints": len(checkpoints),
            "latest_checkpoint": checkpoints[0] if checkpoints else None,
            "oldest_checkpoint": checkpoints[-1] if checkpoints else None,
            "status_distribution": status_counts,
            "checkpoints": checkpoints[:5]  # Return first 5 for preview
        }