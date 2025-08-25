"""Checkpointer implementation using LangGraph PostgresSaver for persistent state management"""

import logging
from typing import Optional
try:
    from langgraph.checkpoint.postgres import PostgresSaver
except ImportError:
    # Fallback to base checkpointer if PostgreSQL checkpoint not available
    from langgraph.checkpoint.memory import MemorySaver as PostgresSaver
from langgraph.checkpoint.base import BaseCheckpointSaver
from src.config.settings import settings

logger = logging.getLogger(__name__)

class SharedCheckpointer:
    """Shared checkpointer using PostgresSaver for persistent state across agents"""
    
    def __init__(self):
        self.checkpointer: Optional[PostgresSaver] = None
        self._initialized = False
    
    def initialize(self):
        """Initialize the PostgresSaver checkpointer"""
        try:
            # Check if we have the PostgreSQL checkpointer available
            if hasattr(PostgresSaver, 'from_conn_string'):
                # Create PostgresSaver with database connection
                import psycopg
                conn = psycopg.connect(settings.database_url)
                self.checkpointer = PostgresSaver(conn=conn)
                
                # Setup database schema for checkpoints (run once)
                self.checkpointer.setup()
                logger.info("Shared checkpointer initialized with PostgresSaver")
            else:
                # Fallback to memory saver
                self.checkpointer = PostgresSaver()
                logger.warning("Using MemorySaver fallback - checkpoints will not persist")
            
            self._initialized = True
            
        except Exception as e:
            logger.error(f"Failed to initialize PostgreSQL checkpointer, falling back to memory: {e}")
            # Fallback to memory checkpointer
            try:
                from langgraph.checkpoint.memory import MemorySaver
                self.checkpointer = MemorySaver()
                self._initialized = True
                logger.warning("Using MemorySaver fallback due to PostgreSQL connection issues")
            except Exception as fallback_error:
                logger.error(f"Failed to initialize fallback checkpointer: {fallback_error}")
                raise
    
    def _ensure_initialized(self):
        """Ensure the checkpointer is initialized"""
        if not self._initialized:
            self.initialize()
    
    def get_checkpointer(self) -> BaseCheckpointSaver:
        """Get the underlying checkpointer for use in LangGraph compilation"""
        self._ensure_initialized()
        return self.checkpointer
    
    def cleanup_old_checkpoints(self, retention_days: Optional[int] = None):
        """Clean up old checkpoints based on retention policy"""
        self._ensure_initialized()
        
        try:
            retention = retention_days or settings.checkpoint_retention_days
            # Note: Actual cleanup implementation depends on PostgresSaver API
            # This is a placeholder for the cleanup logic
            logger.info(f"Checkpoint cleanup policy set to {retention} days")
            
        except Exception as e:
            logger.error(f"Failed to cleanup checkpoints: {e}")

# Global shared checkpointer instance
shared_checkpointer = SharedCheckpointer()

def get_shared_checkpointer() -> SharedCheckpointer:
    """Get the global shared checkpointer instance"""
    return shared_checkpointer