"""Vector operations for RAG functionality"""

import uuid
from typing import List, Dict, Any, Optional, Tuple, Union
from langchain.embeddings import init_embeddings
from langchain_core.documents import Document
from src.utils.supabase_client import get_vector_collection
from src.config.settings import settings
from src.utils.logging import get_logger

logger = get_logger(__name__)

class VectorStore:
    """Vector store operations using vecs/pgvector with LangChain embeddings"""
    
    def __init__(self, collection_name: str = "documents"):
        self.collection_name = collection_name
        self.collection = None
        self.embeddings = None
        self._initialized = False
    
    def initialize(self):
        """Initialize vector collection and embeddings"""
        try:
            # Initialize embeddings
            self.embeddings = init_embeddings(f"openai:{settings.embedding_model}")
            
            # Initialize vector collection
            self.collection = get_vector_collection(
                dimension=settings.vector_dimension,
                collection_name=self.collection_name
            )
            
            self._initialized = True
            logger.info(f"Vector store initialized for collection: {self.collection_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            raise
    
    def _ensure_initialized(self):
        """Ensure the vector store is initialized"""
        if not self._initialized:
            self.initialize()
    
    async def add_documents(self, 
                          documents: List[Union[str, Document]], 
                          metadata: Optional[List[Dict[str, Any]]] = None,
                          embeddings: Optional[List[List[float]]] = None,
                          namespace: str = "default") -> List[str]:
        """Add documents to vector store with embeddings"""
        self._ensure_initialized()
        
        try:
            # Convert string documents to Document objects if needed
            doc_objects = []
            for i, doc in enumerate(documents):
                if isinstance(doc, str):
                    doc_meta = metadata[i] if metadata and i < len(metadata) else {}
                    doc_meta["namespace"] = namespace
                    doc_objects.append(Document(page_content=doc, metadata=doc_meta))
                else:
                    doc.metadata["namespace"] = namespace
                    doc_objects.append(doc)
            
            # Generate embeddings if not provided
            if embeddings is None:
                texts = [doc.page_content for doc in doc_objects]
                embeddings = await self.embeddings.aembed_documents(texts)
            
            # Prepare records for insertion
            records = []
            doc_ids = []
            
            for i, (doc, embedding) in enumerate(zip(doc_objects, embeddings)):
                doc_id = str(uuid.uuid4())
                doc_ids.append(doc_id)
                
                record = {
                    "id": doc_id,
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "embedding": embedding
                }
                records.append(record)
            
            # Insert into vector database
            self.collection.upsert(records)
            
            logger.info(f"Added {len(doc_ids)} documents to collection {self.collection_name}")
            return doc_ids
            
        except Exception as e:
            logger.error(f"Failed to add documents to vector store: {e}")
            raise
    
    async def similarity_search(self, 
                              query: str, 
                              k: int = 5,
                              threshold: float = 0.8,
                              namespace: Optional[str] = None,
                              filter_metadata: Optional[Dict[str, Any]] = None) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search for similar documents using semantic similarity"""
        self._ensure_initialized()
        
        try:
            # Generate query embedding
            query_embedding = await self.embeddings.aembed_query(query)
            
            # Build metadata filter
            filters = {}
            if namespace:
                filters["namespace"] = namespace
            if filter_metadata:
                filters.update(filter_metadata)
            
            # Perform similarity search using vecs API
            results = self.collection.query(
                data=query_embedding,
                limit=k,
                include_metadata=True,
                filters=filters if filters else None
            )
            
            # Process results
            search_results = []
            for result in results:
                content = result.get("content", "")
                score = result.get("distance", 0.0)
                # Convert distance to similarity score (1 - distance for cosine)
                similarity = 1.0 - score if score <= 1.0 else 0.0
                
                if similarity >= threshold:
                    metadata = result.get("metadata", {})
                    search_results.append((content, similarity, metadata))
            
            logger.info(f"Found {len(search_results)} similar documents for query in {self.collection_name}")
            return search_results
            
        except Exception as e:
            logger.error(f"Failed to search vector store: {e}")
            raise
    
    async def delete_documents(self, document_ids: List[str]) -> bool:
        """Delete documents from vector store"""
        self._ensure_initialized()
        
        try:
            # Delete documents by IDs
            self.collection.delete(ids=document_ids)
            
            logger.info(f"Deleted {len(document_ids)} documents from collection {self.collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete documents from vector store: {e}")
            return False
    
    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector collection"""
        self._ensure_initialized()
        
        try:
            # Get collection info (method may vary based on vecs version)
            stats = {
                "collection_name": self.collection_name,
                "dimension": settings.vector_dimension,
                "total_documents": 0  # Placeholder - actual implementation depends on vecs API
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {"error": str(e)}

# Global vector store instances for different collections
vector_stores = {
    "documents": VectorStore("documents"),
    "irs_documents": VectorStore("irs_documents"),
    "financial_regulations": VectorStore("financial_regulations"),
    "tax_guidance": VectorStore("tax_guidance"),
    "user_documents": VectorStore("user_documents")
}

# Default vector store instance (for backward compatibility)
vector_store = vector_stores["documents"]

def get_vector_store(collection_name: str = "documents") -> VectorStore:
    """Get or create a vector store for the specified collection"""
    if collection_name not in vector_stores:
        vector_stores[collection_name] = VectorStore(collection_name)
    
    return vector_stores[collection_name]