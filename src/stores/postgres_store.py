"""PostgreSQL-based store for LangGraph agent memory."""

import json
import logging
from typing import Any, Dict, List, Optional
from uuid import uuid4

import psycopg2
from psycopg2.extras import RealDictCursor

from src.config.settings import settings

logger = logging.getLogger(__name__)


class PostgresStore:
    """PostgreSQL-based store for agent memory and persistent data."""
    
    def __init__(self, connection_string: Optional[str] = None):
        """Initialize the PostgreSQL store.
        
        Args:
            connection_string: PostgreSQL connection string. If None, uses settings.database_url
        """
        self.connection_string = connection_string or settings.database_url
        self._ensure_table_exists()
    
    def _get_connection(self):
        """Get a database connection."""
        return psycopg2.connect(self.connection_string)
    
    def _ensure_table_exists(self):
        """Ensure the agent_memory table exists."""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS agent_memory (
                            id SERIAL PRIMARY KEY,
                            agent_id VARCHAR(255) NOT NULL,
                            memory_key VARCHAR(255) NOT NULL,
                            memory_value JSONB NOT NULL,
                            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                            updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                            UNIQUE(agent_id, memory_key)
                        );
                        
                        CREATE INDEX IF NOT EXISTS idx_agent_memory_agent_id 
                        ON agent_memory(agent_id);
                        
                        CREATE INDEX IF NOT EXISTS idx_agent_memory_created_at 
                        ON agent_memory(created_at DESC);
                    """)
                    conn.commit()
        except Exception as e:
            logger.error(f"Failed to ensure agent_memory table exists: {e}")
            raise
    
    def put(self, agent_id: str, key: str, value: Any) -> None:
        """Store a value in agent memory.
        
        Args:
            agent_id: Unique identifier for the agent
            key: Memory key
            value: Value to store (will be JSON serialized)
        """
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO agent_memory (agent_id, memory_key, memory_value)
                        VALUES (%s, %s, %s)
                        ON CONFLICT (agent_id, memory_key)
                        DO UPDATE SET 
                            memory_value = EXCLUDED.memory_value,
                            updated_at = NOW()
                    """, (agent_id, key, json.dumps(value)))
                    conn.commit()
                    
            logger.debug(f"Stored memory {key} for agent {agent_id}")
            
        except Exception as e:
            logger.error(f"Failed to store memory {key} for agent {agent_id}: {e}")
            raise
    
    def get(self, agent_id: str, key: str, default: Any = None) -> Any:
        """Get a value from agent memory.
        
        Args:
            agent_id: Unique identifier for the agent
            key: Memory key
            default: Default value if key not found
            
        Returns:
            Stored value or default
        """
        try:
            with self._get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("""
                        SELECT memory_value FROM agent_memory
                        WHERE agent_id = %s AND memory_key = %s
                    """, (agent_id, key))
                    
                    row = cur.fetchone()
                    if not row:
                        return default
                    
                    return row["memory_value"]
                    
        except Exception as e:
            logger.error(f"Failed to get memory {key} for agent {agent_id}: {e}")
            return default
    
    def get_all(self, agent_id: str) -> Dict[str, Any]:
        """Get all memory for an agent.
        
        Args:
            agent_id: Unique identifier for the agent
            
        Returns:
            Dictionary of all memory key-value pairs
        """
        try:
            with self._get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("""
                        SELECT memory_key, memory_value FROM agent_memory
                        WHERE agent_id = %s
                        ORDER BY updated_at DESC
                    """, (agent_id,))
                    
                    return {
                        row["memory_key"]: row["memory_value"]
                        for row in cur.fetchall()
                    }
                    
        except Exception as e:
            logger.error(f"Failed to get all memory for agent {agent_id}: {e}")
            return {}
    
    def delete(self, agent_id: str, key: str) -> bool:
        """Delete a memory key for an agent.
        
        Args:
            agent_id: Unique identifier for the agent
            key: Memory key to delete
            
        Returns:
            True if key was deleted, False if not found
        """
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        DELETE FROM agent_memory
                        WHERE agent_id = %s AND memory_key = %s
                    """, (agent_id, key))
                    deleted = cur.rowcount > 0
                    conn.commit()
                    
            if deleted:
                logger.debug(f"Deleted memory {key} for agent {agent_id}")
            
            return deleted
            
        except Exception as e:
            logger.error(f"Failed to delete memory {key} for agent {agent_id}: {e}")
            return False
    
    def delete_all(self, agent_id: str) -> int:
        """Delete all memory for an agent.
        
        Args:
            agent_id: Unique identifier for the agent
            
        Returns:
            Number of memory entries deleted
        """
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        DELETE FROM agent_memory
                        WHERE agent_id = %s
                    """, (agent_id,))
                    deleted_count = cur.rowcount
                    conn.commit()
                    
            logger.debug(f"Deleted {deleted_count} memory entries for agent {agent_id}")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Failed to delete all memory for agent {agent_id}: {e}")
            return 0
    
    def list_agents(self) -> List[str]:
        """List all agent IDs that have memory stored.
        
        Returns:
            List of agent IDs
        """
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT DISTINCT agent_id FROM agent_memory
                        ORDER BY agent_id
                    """)
                    
                    return [row[0] for row in cur.fetchall()]
                    
        except Exception as e:
            logger.error(f"Failed to list agents: {e}")
            return []
    
    def cleanup(self, days_old: int = 30) -> int:
        """Clean up old memory entries.
        
        Args:
            days_old: Delete memory entries older than this many days
            
        Returns:
            Number of memory entries deleted
        """
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        DELETE FROM agent_memory
                        WHERE updated_at < NOW() - INTERVAL '%s days'
                    """, (days_old,))
                    deleted_count = cur.rowcount
                    conn.commit()
                    
            logger.info(f"Cleaned up {deleted_count} old memory entries")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup old memory entries: {e}")
            raise