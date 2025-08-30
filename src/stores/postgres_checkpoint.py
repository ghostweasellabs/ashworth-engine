"""PostgreSQL-based checkpointer for LangGraph workflows."""

import json
import logging
from typing import Any, Dict, Iterator, Optional, Tuple
from uuid import uuid4

import psycopg2
from langgraph.checkpoint.base import BaseCheckpointSaver, Checkpoint, CheckpointMetadata
from psycopg2.extras import RealDictCursor

from src.config.settings import settings

logger = logging.getLogger(__name__)


class PostgresCheckpointer(BaseCheckpointSaver):
    """PostgreSQL-based checkpoint saver for LangGraph workflows."""
    
    def __init__(self, connection_string: Optional[str] = None):
        """Initialize the PostgreSQL checkpointer.
        
        Args:
            connection_string: PostgreSQL connection string. If None, uses settings.database_url
        """
        self.connection_string = connection_string or settings.database_url
        self._ensure_table_exists()
    
    def _get_connection(self):
        """Get a database connection."""
        return psycopg2.connect(self.connection_string)
    
    def _ensure_table_exists(self):
        """Ensure the checkpoints table exists."""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS workflow_checkpoints (
                            id SERIAL PRIMARY KEY,
                            workflow_id VARCHAR(255) NOT NULL,
                            checkpoint_id VARCHAR(255) NOT NULL,
                            state JSONB NOT NULL,
                            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                            updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                            UNIQUE(workflow_id, checkpoint_id)
                        );
                        
                        CREATE INDEX IF NOT EXISTS idx_workflow_checkpoints_workflow_id 
                        ON workflow_checkpoints(workflow_id);
                        
                        CREATE INDEX IF NOT EXISTS idx_workflow_checkpoints_created_at 
                        ON workflow_checkpoints(created_at DESC);
                    """)
                    conn.commit()
        except Exception as e:
            logger.error(f"Failed to ensure checkpoints table exists: {e}")
            raise
    
    def put(
        self,
        config: Dict[str, Any],
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
    ) -> None:
        """Save a checkpoint to PostgreSQL.
        
        Args:
            config: Configuration dictionary containing workflow_id
            checkpoint: The checkpoint data to save
            metadata: Checkpoint metadata
        """
        workflow_id = config.get("configurable", {}).get("thread_id", str(uuid4()))
        checkpoint_id = checkpoint.get("id", str(uuid4()))
        
        checkpoint_data = {
            "checkpoint": checkpoint,
            "metadata": metadata,
            "config": config
        }
        
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO workflow_checkpoints (workflow_id, checkpoint_id, state)
                        VALUES (%s, %s, %s)
                        ON CONFLICT (workflow_id, checkpoint_id)
                        DO UPDATE SET 
                            state = EXCLUDED.state,
                            updated_at = NOW()
                    """, (workflow_id, checkpoint_id, json.dumps(checkpoint_data)))
                    conn.commit()
                    
            logger.debug(f"Saved checkpoint {checkpoint_id} for workflow {workflow_id}")
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint {checkpoint_id}: {e}")
            raise
    
    def get(
        self,
        config: Dict[str, Any]
    ) -> Optional[Tuple[Checkpoint, CheckpointMetadata]]:
        """Get the latest checkpoint for a workflow.
        
        Args:
            config: Configuration dictionary containing workflow_id
            
        Returns:
            Tuple of (checkpoint, metadata) or None if not found
        """
        workflow_id = config.get("configurable", {}).get("thread_id")
        if not workflow_id:
            return None
            
        try:
            with self._get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("""
                        SELECT state FROM workflow_checkpoints
                        WHERE workflow_id = %s
                        ORDER BY created_at DESC
                        LIMIT 1
                    """, (workflow_id,))
                    
                    row = cur.fetchone()
                    if not row:
                        return None
                    
                    checkpoint_data = row["state"]
                    return (
                        checkpoint_data["checkpoint"],
                        checkpoint_data["metadata"]
                    )
                    
        except Exception as e:
            logger.error(f"Failed to get checkpoint for workflow {workflow_id}: {e}")
            return None
    
    def list(
        self,
        config: Dict[str, Any],
        limit: Optional[int] = None,
        before: Optional[str] = None
    ) -> Iterator[Tuple[Checkpoint, CheckpointMetadata]]:
        """List checkpoints for a workflow.
        
        Args:
            config: Configuration dictionary containing workflow_id
            limit: Maximum number of checkpoints to return
            before: Return checkpoints before this checkpoint_id
            
        Yields:
            Tuples of (checkpoint, metadata)
        """
        workflow_id = config.get("configurable", {}).get("thread_id")
        if not workflow_id:
            return
            
        try:
            with self._get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    query = """
                        SELECT state FROM workflow_checkpoints
                        WHERE workflow_id = %s
                    """
                    params = [workflow_id]
                    
                    if before:
                        query += " AND checkpoint_id < %s"
                        params.append(before)
                    
                    query += " ORDER BY created_at DESC"
                    
                    if limit:
                        query += " LIMIT %s"
                        params.append(limit)
                    
                    cur.execute(query, params)
                    
                    for row in cur.fetchall():
                        checkpoint_data = row["state"]
                        yield (
                            checkpoint_data["checkpoint"],
                            checkpoint_data["metadata"]
                        )
                        
        except Exception as e:
            logger.error(f"Failed to list checkpoints for workflow {workflow_id}: {e}")
    
    def delete(self, config: Dict[str, Any]) -> None:
        """Delete all checkpoints for a workflow.
        
        Args:
            config: Configuration dictionary containing workflow_id
        """
        workflow_id = config.get("configurable", {}).get("thread_id")
        if not workflow_id:
            return
            
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        DELETE FROM workflow_checkpoints
                        WHERE workflow_id = %s
                    """, (workflow_id,))
                    conn.commit()
                    
            logger.debug(f"Deleted checkpoints for workflow {workflow_id}")
            
        except Exception as e:
            logger.error(f"Failed to delete checkpoints for workflow {workflow_id}: {e}")
            raise
    
    def cleanup(self, days_old: int = 30) -> int:
        """Clean up old checkpoints.
        
        Args:
            days_old: Delete checkpoints older than this many days
            
        Returns:
            Number of checkpoints deleted
        """
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        DELETE FROM workflow_checkpoints
                        WHERE created_at < NOW() - INTERVAL '%s days'
                    """, (days_old,))
                    deleted_count = cur.rowcount
                    conn.commit()
                    
            logger.info(f"Cleaned up {deleted_count} old checkpoints")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup old checkpoints: {e}")
            raise