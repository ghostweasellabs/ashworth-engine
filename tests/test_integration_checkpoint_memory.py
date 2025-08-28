"""Integration tests for checkpoint and memory functionality with real database."""

import pytest
import os
from datetime import datetime
from uuid import uuid4

# Set test environment before importing modules
os.environ["DATABASE_URL"] = "postgresql://postgres:postgres@127.0.0.1:54322/postgres"

from src.stores.postgres_checkpoint import PostgresCheckpointer
from src.stores.postgres_store import PostgresStore
from src.utils.checkpoint_manager import CheckpointManager
from src.workflows.state_schemas import (
    create_initial_workflow_state,
    update_agent_state,
    WorkflowStatus,
    AgentStatus
)


class TestIntegrationCheckpointMemory:
    """Integration tests with real database connection."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.workflow_id = f"test-workflow-{uuid4().hex[:8]}"
        self.checkpointer = PostgresCheckpointer()
        self.store = PostgresStore()
        self.manager = CheckpointManager(self.checkpointer)
    
    def test_database_connection(self):
        """Test that we can connect to the database."""
        # Test checkpointer connection
        try:
            with self.checkpointer._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    result = cur.fetchone()
                    assert result[0] == 1
        except Exception as e:
            pytest.fail(f"Failed to connect to database: {e}")
    
    def test_checkpoint_full_workflow(self):
        """Test complete checkpoint workflow with real database."""
        # Create initial workflow state
        state = create_initial_workflow_state(
            self.workflow_id,
            workflow_type="financial_analysis",
            input_files=["test1.xlsx", "test2.csv"]
        )
        
        # Create initial checkpoint
        checkpoint_id = self.manager.create_checkpoint(
            self.workflow_id,
            state,
            checkpoint_name="initial_state",
            metadata={"test": True, "step": 0}
        )
        
        assert checkpoint_id.startswith(self.workflow_id)
        
        # Update state to simulate agent progress
        state = update_agent_state(
            state,
            "data_fetcher",
            AgentStatus.IN_PROGRESS
        )
        
        # Create progress checkpoint
        progress_checkpoint_id = self.manager.create_checkpoint(
            self.workflow_id,
            state,
            checkpoint_name="data_fetcher_started",
            metadata={"test": True, "step": 1}
        )
        
        # Complete the agent
        state = update_agent_state(
            state,
            "data_fetcher",
            AgentStatus.COMPLETED,
            output_data={"processed_files": ["test1.xlsx", "test2.csv"]}
        )
        
        # Create completion checkpoint
        completion_checkpoint_id = self.manager.create_checkpoint(
            self.workflow_id,
            state,
            checkpoint_name="data_fetcher_completed",
            metadata={"test": True, "step": 2}
        )
        
        # List checkpoints
        checkpoints = self.manager.list_checkpoints(self.workflow_id)
        assert len(checkpoints) >= 3
        
        # Verify checkpoint order (newest first)
        assert checkpoints[0]["name"] == "data_fetcher_completed"
        assert checkpoints[1]["name"] == "data_fetcher_started"
        assert checkpoints[2]["name"] == "initial_state"
        
        # Test rollback functionality
        restored_state = self.manager.rollback_to_checkpoint(
            self.workflow_id,
            progress_checkpoint_id
        )
        
        assert restored_state is not None
        assert "rollback_from" in restored_state["checkpoint_metadata"]
        assert restored_state["checkpoint_metadata"]["rollback_from"] == progress_checkpoint_id
    
    def test_agent_memory_full_workflow(self):
        """Test complete agent memory workflow with real database."""
        agent_id = f"test-agent-{uuid4().hex[:8]}"
        
        # Store various types of memory
        self.store.put(agent_id, "last_processed_file", "test1.xlsx")
        self.store.put(agent_id, "error_count", 0)
        self.store.put(agent_id, "processing_stats", {
            "files_processed": 5,
            "total_size": 1024000,
            "avg_processing_time": 2.5
        })
        self.store.put(agent_id, "last_run_timestamp", datetime.utcnow().isoformat())
        
        # Retrieve individual memories
        last_file = self.store.get(agent_id, "last_processed_file")
        assert last_file == "test1.xlsx"
        
        error_count = self.store.get(agent_id, "error_count")
        assert error_count == 0
        
        stats = self.store.get(agent_id, "processing_stats")
        assert stats["files_processed"] == 5
        assert stats["total_size"] == 1024000
        
        # Test default value
        missing_key = self.store.get(agent_id, "nonexistent_key", "default_value")
        assert missing_key == "default_value"
        
        # Get all memory
        all_memory = self.store.get_all(agent_id)
        assert len(all_memory) == 4
        assert "last_processed_file" in all_memory
        assert "error_count" in all_memory
        assert "processing_stats" in all_memory
        assert "last_run_timestamp" in all_memory
        
        # Update existing memory
        self.store.put(agent_id, "error_count", 1)
        updated_count = self.store.get(agent_id, "error_count")
        assert updated_count == 1
        
        # Delete specific memory
        deleted = self.store.delete(agent_id, "last_run_timestamp")
        assert deleted is True
        
        # Verify deletion
        remaining_memory = self.store.get_all(agent_id)
        assert len(remaining_memory) == 3
        assert "last_run_timestamp" not in remaining_memory
        
        # Delete all memory
        deleted_count = self.store.delete_all(agent_id)
        assert deleted_count == 3
        
        # Verify all deleted
        final_memory = self.store.get_all(agent_id)
        assert len(final_memory) == 0
    
    def test_checkpoint_stats(self):
        """Test checkpoint statistics functionality."""
        # Create workflow with multiple checkpoints
        state = create_initial_workflow_state(self.workflow_id)
        
        # Create checkpoints with different statuses
        self.manager.create_checkpoint(
            self.workflow_id,
            state,
            checkpoint_name="initial",
            metadata={"step": 0}
        )
        
        state["status"] = WorkflowStatus.RUNNING
        self.manager.create_checkpoint(
            self.workflow_id,
            state,
            checkpoint_name="running",
            metadata={"step": 1}
        )
        
        state["status"] = WorkflowStatus.COMPLETED
        self.manager.create_checkpoint(
            self.workflow_id,
            state,
            checkpoint_name="completed",
            metadata={"step": 2}
        )
        
        # Get statistics
        stats = self.manager.get_checkpoint_stats(self.workflow_id)
        
        assert stats["total_checkpoints"] >= 3
        assert stats["latest_checkpoint"]["name"] == "completed"
        assert "status_distribution" in stats
        assert WorkflowStatus.COMPLETED in stats["status_distribution"]
        assert len(stats["checkpoints"]) <= 5  # Preview limit
    
    def test_list_agents_with_memory(self):
        """Test listing agents that have memory stored."""
        # Create memory for multiple agents
        agent1 = f"agent-1-{uuid4().hex[:8]}"
        agent2 = f"agent-2-{uuid4().hex[:8]}"
        agent3 = f"agent-3-{uuid4().hex[:8]}"
        
        self.store.put(agent1, "test_key", "test_value1")
        self.store.put(agent2, "test_key", "test_value2")
        self.store.put(agent3, "test_key", "test_value3")
        
        # List all agents
        agents = self.store.list_agents()
        
        assert agent1 in agents
        assert agent2 in agents
        assert agent3 in agents
        assert len(agents) >= 3
    
    def teardown_method(self):
        """Clean up test data."""
        try:
            # Clean up checkpoints
            config = {
                "configurable": {
                    "thread_id": self.workflow_id
                }
            }
            self.checkpointer.delete(config)
        except Exception:
            pass  # Ignore cleanup errors


if __name__ == "__main__":
    # Simple test runner for development
    test_instance = TestIntegrationCheckpointMemory()
    
    print("Setting up test...")
    test_instance.setup_method()
    
    try:
        print("Testing database connection...")
        test_instance.test_database_connection()
        print("✅ Database connection test passed")
        
        print("\nTesting checkpoint workflow...")
        test_instance.test_checkpoint_full_workflow()
        print("✅ Checkpoint workflow test passed")
        
        print("\nTesting agent memory workflow...")
        test_instance.test_agent_memory_full_workflow()
        print("✅ Agent memory workflow test passed")
        
        print("\nTesting checkpoint statistics...")
        test_instance.test_checkpoint_stats()
        print("✅ Checkpoint statistics test passed")
        
        print("\nTesting agent listing...")
        test_instance.test_list_agents_with_memory()
        print("✅ Agent listing test passed")
        
    finally:
        print("\nCleaning up...")
        test_instance.teardown_method()
    
    print("\n🎉 All integration tests completed successfully!")