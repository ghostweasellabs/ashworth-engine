"""
Vector store implementation using Supabase PostgreSQL with pgvector.
Handles storing and retrieving document embeddings with similarity search.
"""

import logging
from typing import Dict, List, Optional, Tuple

import asyncpg
from pydantic import BaseModel

from src.config.settings import get_settings

logger = logging.getLogger(__name__)


class SearchResult(BaseModel):
    """Result from vector similarity search."""
    content: str
    similarity_score: float
    metadata: Dict
    document_id: str
    chunk_index: int


class VectorStore:
    """Vector store using PostgreSQL with pgvector extension."""
    
    def __init__(self):
        self.settings = get_settings()
        self.connection_pool = None
        
    async def _get_connection(self) -> asyncpg.Connection:
        """Get database connection."""
        if not self.connection_pool:
            self.connection_pool = await asyncpg.create_pool(
                self.settings.database_url,
                min_size=1,
                max_size=10
            )
        
        return await self.connection_pool.acquire()
    
    async def _release_connection(self, conn: asyncpg.Connection):
        """Release database connection."""
        if self.connection_pool:
            await self.connection_pool.release(conn)
    
    async def store_source_document(
        self,
        document_id: str,
        title: str,
        source_url: Optional[str],
        document_type: str,
        content: str,
        metadata: Dict
    ) -> bool:
        """
        Store source document in the database.
        
        Args:
            document_id: Unique identifier for the document
            title: Document title
            source_url: URL where document was sourced from
            document_type: Type of document (e.g., 'irs_publication')
            content: Full document content
            metadata: Additional metadata
            
        Returns:
            True if successful, False otherwise
        """
        conn = None
        try:
            conn = await self._get_connection()
            
            import json
            metadata_json = json.dumps(metadata) if isinstance(metadata, dict) else metadata
            
            await conn.execute("""
                INSERT INTO source_documents (document_id, title, source_url, document_type, content, metadata)
                VALUES ($1, $2, $3, $4, $5, $6::jsonb)
                ON CONFLICT (document_id) 
                DO UPDATE SET 
                    title = EXCLUDED.title,
                    source_url = EXCLUDED.source_url,
                    document_type = EXCLUDED.document_type,
                    content = EXCLUDED.content,
                    metadata = EXCLUDED.metadata,
                    updated_at = NOW()
            """, document_id, title, source_url, document_type, content, metadata_json)
            
            logger.info(f"Stored source document: {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing source document {document_id}: {e}")
            return False
        finally:
            if conn:
                await self._release_connection(conn)
    
    async def store_embeddings(self, document_id: str, chunks: List) -> bool:
        """
        Store document chunk embeddings in the vector database.
        
        Args:
            document_id: Unique identifier for the document
            chunks: List of TextChunk objects with embeddings
            
        Returns:
            True if successful, False otherwise
        """
        conn = None
        try:
            conn = await self._get_connection()
            
            # Delete existing embeddings for this document
            await conn.execute(
                "DELETE FROM document_embeddings WHERE document_id = $1",
                document_id
            )
            
            # Insert new embeddings
            for chunk in chunks:
                if chunk.embedding is None:
                    logger.warning(f"Skipping chunk {chunk.chunk_index} - no embedding")
                    continue
                
                import json
                metadata_json = json.dumps(chunk.metadata) if isinstance(chunk.metadata, dict) else chunk.metadata
                
                # Convert embedding list to string format for pgvector
                embedding_str = '[' + ','.join(map(str, chunk.embedding)) + ']'
                
                await conn.execute("""
                    INSERT INTO document_embeddings 
                    (document_id, chunk_index, content, metadata, embedding)
                    VALUES ($1, $2, $3, $4::jsonb, $5::vector)
                """, 
                document_id, 
                chunk.chunk_index, 
                chunk.content, 
                metadata_json, 
                embedding_str
                )
            
            logger.info(f"Stored {len(chunks)} embeddings for document {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing embeddings for {document_id}: {e}")
            return False
        finally:
            if conn:
                await self._release_connection(conn)
    
    async def similarity_search(
        self,
        query_embedding: List[float],
        limit: int = 10,
        similarity_threshold: float = 0.7,
        document_type: Optional[str] = None,
        metadata_filter: Optional[Dict] = None
    ) -> List[SearchResult]:
        """
        Perform vector similarity search.
        
        Args:
            query_embedding: Query vector
            limit: Maximum number of results
            similarity_threshold: Minimum similarity score (0-1)
            document_type: Filter by document type
            metadata_filter: Additional metadata filters
            
        Returns:
            List of SearchResult objects
        """
        conn = None
        try:
            conn = await self._get_connection()
            
            # Convert query embedding to string format for pgvector
            query_embedding_str = '[' + ','.join(map(str, query_embedding)) + ']'
            
            # Build query with optional filters
            base_query = """
                SELECT 
                    de.content,
                    de.metadata,
                    de.document_id,
                    de.chunk_index,
                    1 - (de.embedding <=> $1::vector) as similarity_score
                FROM document_embeddings de
                JOIN source_documents sd ON de.document_id = sd.document_id
                WHERE 1 - (de.embedding <=> $1::vector) >= $2
            """
            
            params = [query_embedding_str, similarity_threshold]
            param_count = 2
            
            # Add document type filter
            if document_type:
                param_count += 1
                base_query += f" AND sd.document_type = ${param_count}"
                params.append(document_type)
            
            # Add metadata filters
            if metadata_filter:
                for key, value in metadata_filter.items():
                    param_count += 1
                    base_query += f" AND de.metadata->>'{key}' = ${param_count}"
                    params.append(str(value))
            
            # Order by similarity and limit
            base_query += f" ORDER BY similarity_score DESC LIMIT ${param_count + 1}"
            params.append(limit)
            
            rows = await conn.fetch(base_query, *params)
            
            results = []
            for row in rows:
                # Parse metadata if it's a string
                metadata = row['metadata']
                if isinstance(metadata, str):
                    import json
                    try:
                        metadata = json.loads(metadata)
                    except json.JSONDecodeError:
                        metadata = {}
                
                result = SearchResult(
                    content=row['content'],
                    similarity_score=float(row['similarity_score']),
                    metadata=metadata,
                    document_id=row['document_id'],
                    chunk_index=row['chunk_index']
                )
                results.append(result)
            
            logger.info(f"Found {len(results)} similar chunks with threshold {similarity_threshold}")
            return results
            
        except Exception as e:
            logger.error(f"Error performing similarity search: {e}")
            return []
        finally:
            if conn:
                await self._release_connection(conn)
    
    async def get_document_chunks(self, document_id: str) -> List[Dict]:
        """Get all chunks for a specific document."""
        conn = None
        try:
            conn = await self._get_connection()
            
            rows = await conn.fetch("""
                SELECT content, metadata, chunk_index
                FROM document_embeddings
                WHERE document_id = $1
                ORDER BY chunk_index
            """, document_id)
            
            return [dict(row) for row in rows]
            
        except Exception as e:
            logger.error(f"Error getting chunks for document {document_id}: {e}")
            return []
        finally:
            if conn:
                await self._release_connection(conn)
    
    async def get_ingestion_stats(self) -> Dict:
        """Get statistics about ingested documents."""
        conn = None
        try:
            conn = await self._get_connection()
            
            # Get document counts by type
            doc_stats = await conn.fetch("""
                SELECT document_type, COUNT(*) as count
                FROM source_documents
                GROUP BY document_type
            """)
            
            # Get total chunk count
            chunk_count = await conn.fetchval("""
                SELECT COUNT(*) FROM document_embeddings
            """)
            
            # Get total document count
            doc_count = await conn.fetchval("""
                SELECT COUNT(*) FROM source_documents
            """)
            
            stats = {
                'total_documents': doc_count or 0,
                'total_chunks': chunk_count or 0,
                'documents_by_type': {row['document_type']: row['count'] for row in doc_stats}
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting ingestion stats: {e}")
            return {}
        finally:
            if conn:
                await self._release_connection(conn)
    
    async def delete_document(self, document_id: str) -> bool:
        """Delete a document and all its embeddings."""
        conn = None
        try:
            conn = await self._get_connection()
            
            # Delete embeddings first
            await conn.execute(
                "DELETE FROM document_embeddings WHERE document_id = $1",
                document_id
            )
            
            # Delete source document
            await conn.execute(
                "DELETE FROM source_documents WHERE document_id = $1",
                document_id
            )
            
            logger.info(f"Deleted document: {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting document {document_id}: {e}")
            return False
        finally:
            if conn:
                await self._release_connection(conn)
    
    async def close(self):
        """Close the connection pool."""
        if self.connection_pool:
            await self.connection_pool.close()


# Test function
async def test_vector_store():
    """Test vector store functionality."""
    store = VectorStore()
    
    # Test storing a document
    success = await store.store_source_document(
        document_id="test_doc",
        title="Test Document",
        source_url=None,
        document_type="test",
        content="This is a test document for vector store testing.",
        metadata={"test": True}
    )
    
    if success:
        print("Successfully stored test document")
        
        # Get stats
        stats = await store.get_ingestion_stats()
        print(f"Ingestion stats: {stats}")
    else:
        print("Failed to store test document")
    
    await store.close()


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_vector_store())