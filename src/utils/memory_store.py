"""Memory store implementation using LangGraph PostgresStore for shared memory across agents"""

import uuid
from typing import Dict, Any, List, Optional, Tuple, Union
from langchain.embeddings import init_embeddings
from langgraph.store.postgres import PostgresStore
from langgraph.store.base import BaseStore
from src.config.settings import settings
from src.utils.logging import get_logger

logger = get_logger(__name__)

class SharedMemoryStore:
    """Shared memory store using PostgresStore with semantic search capabilities"""
    
    def __init__(self):
        self.store: Optional[PostgresStore] = None
        self.embeddings = None
        self._initialized = False
    
    def initialize(self):
        """Initialize the PostgresStore with embedding configuration"""
        try:
            # Initialize embeddings only if API key is available
            if settings.openai_api_key:
                self.embeddings = init_embeddings(f"openai:{settings.embedding_model}")
                
                # Initialize PostgresStore with index configuration for semantic search
                index_config = {
                    "dims": settings.vector_dimension,
                    "embed": self.embeddings,
                    "fields": settings.store_index_config["fields"]
                }
                
                # Import psycopg for connection
                import psycopg
                conn = psycopg.connect(settings.database_url)
                
                self.store = PostgresStore(
                    conn=conn,
                    index=index_config
                )
            else:
                logger.warning("OpenAI API key not found, initializing PostgresStore without embeddings")
                # Initialize PostgresStore without embeddings for basic storage
                import psycopg
                conn = psycopg.connect(settings.database_url)
                self.store = PostgresStore(conn=conn)
            
            # Setup database schema (run once) - only call if method exists
            if hasattr(self.store, 'setup'):
                self.store.setup()
            
            self._initialized = True
            logger.info("Shared memory store initialized with PostgresStore")
            
        except Exception as e:
            logger.error(f"Failed to initialize shared memory store: {e}")
            raise
    
    def _ensure_initialized(self):
        """Ensure the store is initialized"""
        if not self._initialized:
            self.initialize()
    
    async def put_memory(self, 
                        namespace: Tuple[str, ...], 
                        key: str, 
                        value: Dict[str, Any], 
                        index: Union[bool, List[str], None] = None) -> str:
        """Store a memory item in the shared store"""
        self._ensure_initialized()
        
        try:
            # Generate unique key if not provided
            if not key:
                key = str(uuid.uuid4())
            
            # Store the memory
            await self.store.aput(namespace, key, value, index=index)
            
            logger.debug(f"Stored memory with key {key} in namespace {namespace}")
            return key
            
        except Exception as e:
            logger.error(f"Failed to store memory: {e}")
            raise
    
    async def get_memory(self, namespace: Tuple[str, ...], key: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific memory item"""
        self._ensure_initialized()
        
        try:
            item = await self.store.aget(namespace, key)
            return item.value if item else None
            
        except Exception as e:
            logger.error(f"Failed to retrieve memory: {e}")
            return None
    
    async def search_memories(self, 
                            namespace: Tuple[str, ...], 
                            query: Optional[str] = None,
                            filter_dict: Optional[Dict[str, Any]] = None,
                            limit: int = 10) -> List[Dict[str, Any]]:
        """Search for memories using semantic search or metadata filtering"""
        self._ensure_initialized()
        
        try:
            search_results = await self.store.asearch(
                namespace,
                query=query,
                filter=filter_dict,
                limit=limit
            )
            
            # Convert search results to dictionaries
            memories = []
            for item in search_results:
                memory = {
                    "key": item.key,
                    "value": item.value,
                    "namespace": item.namespace,
                    "created_at": item.created_at,
                    "updated_at": item.updated_at
                }
                memories.append(memory)
            
            logger.debug(f"Found {len(memories)} memories for namespace {namespace}")
            return memories
            
        except Exception as e:
            logger.error(f"Failed to search memories: {e}")
            return []
    
    async def delete_memory(self, namespace: Tuple[str, ...], key: str) -> bool:
        """Delete a specific memory item"""
        self._ensure_initialized()
        
        try:
            await self.store.adelete(namespace, key)
            logger.debug(f"Deleted memory with key {key} from namespace {namespace}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete memory: {e}")
            return False
    
    async def list_memories(self, 
                          namespace: Tuple[str, ...], 
                          limit: int = 100,
                          offset: int = 0) -> List[Dict[str, Any]]:
        """List all memories in a namespace"""
        self._ensure_initialized()
        
        try:
            # Use search without query to get all items
            search_results = await self.store.asearch(
                namespace,
                limit=limit,
                offset=offset
            )
            
            memories = []
            for item in search_results:
                memory = {
                    "key": item.key,
                    "value": item.value,
                    "namespace": item.namespace,
                    "created_at": item.created_at,
                    "updated_at": item.updated_at
                }
                memories.append(memory)
            
            return memories
            
        except Exception as e:
            logger.error(f"Failed to list memories: {e}")
            return []
    
    def get_store(self) -> BaseStore:
        """Get the underlying BaseStore instance for direct use in LangGraph"""
        self._ensure_initialized()
        return self.store

# Global shared memory store instance
shared_memory_store = SharedMemoryStore()

# Memory namespace constants for different types of shared data
class MemoryNamespaces:
    """Constants for memory namespaces"""
    
    # Agent-specific memories
    AGENT_MEMORIES = ("agents", "memories")
    TAX_CATEGORIZER_MEMORIES = ("agents", "tax_categorizer", "memories")
    REPORT_GENERATOR_MEMORIES = ("agents", "report_generator", "memories")
    REPORT_GENERATOR_MEMORIES = ("agents", "report_generator", "memories")
    
    # Workflow-specific memories  
    WORKFLOW_STATE = ("workflow", "state")
    FINANCIAL_ANALYSIS = ("workflow", "financial_analysis")
    
    # User-specific memories
    USER_PREFERENCES = ("users", "preferences")
    USER_CONTEXT = ("users", "context")
    
    # System-wide memories
    SYSTEM_CONFIG = ("system", "config")
    IRS_GUIDANCE = ("system", "irs_guidance")
    FINANCIAL_RULES = ("system", "financial_rules")
    
    @staticmethod
    def user_namespace(user_id: str) -> Tuple[str, ...]:
        """Generate user-specific namespace"""
        return ("users", user_id)
    
    @staticmethod
    def agent_namespace(agent_name: str, user_id: Optional[str] = None) -> Tuple[str, ...]:
        """Generate agent-specific namespace, optionally user-scoped"""
        if user_id:
            return ("agents", agent_name, user_id)
        return ("agents", agent_name)
    
    @staticmethod
    def workflow_namespace(workflow_name: str, thread_id: Optional[str] = None) -> Tuple[str, ...]:
        """Generate workflow-specific namespace, optionally thread-scoped"""
        if thread_id:
            return ("workflows", workflow_name, thread_id)
        return ("workflows", workflow_name)

def get_shared_memory_store() -> SharedMemoryStore:
    """Get the global shared memory store instance"""
    return shared_memory_store