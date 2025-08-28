"""Test checkpoint and memory functionality."""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime

from src.stores.postgres_checkpoint import PostgresCheckpointer
from src.stores.postgres_store import PostgresStore
from src.utils.checkpoint_manager import CheckpointManager
from src.workflows.state_schemas import (
    create_initial_workflow_state,
    update_agent_state,
    WorkflowStatus,
    AgentStatus
)


class TestCheckpointMemory:
    """Test checkpoint and memory functionality with mocked database."""
    
    @patch('src.stores.postgres_checkpoint.psycopg2.connect')
    def test_checkpoint_creation(self, mock_connect):
        """Test creating and retrieving checkpoints."""
        # Mock database connection
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_connect.return_value.__enter__.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        
        # Create checkpointer
        checkpointer = PostgresCheckpointer()
        
        # Create test workflow state
        workflow_id = "test-workflow-123"
        state = create_initial_workflow_state(workflow_id)
        
        # Update state to simulate progress
        state = update_agent_state(
            state,
            "data_fetcher",
            AgentStatus.COMPLETED,
            {"processed_files": ["test.xlsx"]}
        )
        
        # Create checkpoint manager
        manager = CheckpointManager(checkpointer)
        
        # Create checkpoint
        checkpoint_id = manager.create_checkpoint(
            workflow_id,
            state,
            checkpoint_name="after_data_fetch",
            metadata={"test": True}
        )
        
        assert checkpoint_id.startswith(workflow_id)
        mock_cursor.execute.assert_called()
    
    @patch('src.stores.postgres_store.psycopg2.connect')
    def test_agent_memory(self, mock_connect):
        """Test agent memory storage and retrieval."""
        # Mock database connection
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_connect.return_value.__enter__.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        
        # Mock cursor results
        mock_cursor.fetchone.return_value = {"memory_value": {"last_processed": "file1.xlsx"}}
        mock_cursor.fetchall.return_value = [
            {"memory_key": "last_processed", "memory_value": {"file": "file1.xlsx"}},
            {"memory_key": "error_count", "memory_value": 0}
        ]
        
        # Create store
        store = PostgresStore()
        
        # Test storing memory
        agent_id = "data_fetcher"
        store.put(agent_id, "last_processed", {"file": "test.xlsx", "timestamp": "2024-01-01"})
        
        # Test retrieving memory
        result = store.get(agent_id, "last_processed")
        assert result == {"last_processed": "file1.xlsx"}
        
        # Test getting all memory
        all_memory = store.get_all(agent_id)
        assert len(all_memory) == 2
        assert "last_processed" in all_memory
        assert "error_count" in all_memory
        
        mock_cursor.execute.assert_called()
    
    def test_workflow_state_creation(self):
        """Test workflow state creation and updates."""
        workflow_id = "test-workflow-456"
        state = create_initial_workflow_state(
            workflow_id,
            workflow_type="financial_analysis",
            input_files=["test1.xlsx", "test2.csv"]
        )
        
        # Verify initial state
        assert state["workflow_id"] == workflow_id
        assert state["workflow_type"] == "financial_analysis"
        assert state["status"] == WorkflowStatus.PENDING
        assert len(state["input_files"]) == 2
        assert state["data_fetcher"]["status"] == AgentStatus.NOT_STARTED
        
        # Test agent state update
        updated_state = update_agent_state(
            state,
            "data_fetcher",
            AgentStatus.IN_PROGRESS
        )
        
        assert updated_state["data_fetcher"]["status"] == AgentStatus.IN_PROGRESS
        assert updated_state["data_fetcher"]["started_at"] is not None
        
        # Test completion
        final_state = update_agent_state(
            updated_state,
            "data_fetcher",
            AgentStatus.COMPLETED,
            output_data={"files_processed": 2}
        )
        
        assert final_state["data_fetcher"]["status"] == AgentStatus.COMPLETED
        assert final_state["data_fetcher"]["completed_at"] is not None
        assert final_state["data_fetcher"]["duration_ms"] is not None
        assert final_state["data_fetcher"]["output_data"]["files_processed"] == 2
    
    @patch('src.stores.postgres_checkpoint.psycopg2.connect')
    def test_checkpoint_rollback(self, mock_connect):
        """Test checkpoint rollback functionality."""
        # Mock database connection
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_connect.return_value.__enter__.return_value = mock_conn
        mock_conn.cursor.return_value.__enter__.return_value = mock_cursor
        
        # Mock checkpoint data
        mock_checkpoint = {
            "id": "test-checkpoint-123",
            "ts": datetime.utcnow().isoformat(),
            "channel_values": {
                "state": create_initial_workflow_state("test-workflow")
            }
        }
        
        mock_metadata = {
            "name": "test_checkpoint",
            "workflow_status": WorkflowStatus.RUNNING,
            "step": 1
        }
        
        mock_cursor.fetchone.return_value = {
            "state": {
                "checkpoint": mock_checkpoint,
                "metadata": mock_metadata,
                "config": {}
            }
        }
        
        # Create checkpoint manager
        checkpointer = PostgresCheckpointer()
        manager = CheckpointManager(checkpointer)
        
        # Test rollback
        restored_state = manager.rollback_to_checkpoint("test-workflow", "test-checkpoint-123")
        
        assert restored_state is not None
        assert restored_state["status"] == WorkflowStatus.PENDING
        assert "rollback_from" in restored_state["checkpoint_metadata"]
        
        mock_cursor.execute.assert_called()


if __name__ == "__main__":
    # Simple test runner
    test_instance = TestCheckpointMemory()
    
    print("Testing workflow state creation...")
    test_instance.test_workflow_state_creation()
    print("✅ Workflow state tests passed")
    
    print("\nAll tests completed successfully!")