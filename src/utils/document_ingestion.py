"""Document ingestion system for RAG functionality"""

import os
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Iterator
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    Docx2txtLoader,
    CSVLoader,
    UnstructuredExcelLoader
)
from src.utils.vector_operations import get_vector_store
from src.utils.memory_store import get_shared_memory_store, MemoryNamespaces
from src.config.settings import settings
from src.utils.logging import get_logger

logger = get_logger(__name__)

class DocumentIngestion:
    """Document ingestion system for RAG with chunking and metadata extraction"""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.rag_chunk_size,
            chunk_overlap=settings.rag_chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        self.supported_extensions = {
            '.pdf': PyPDFLoader,
            '.txt': TextLoader,
            '.md': UnstructuredMarkdownLoader,
            '.docx': Docx2txtLoader,
            '.csv': CSVLoader,
            '.xlsx': UnstructuredExcelLoader,
            '.xls': UnstructuredExcelLoader
        }
    
    def _get_loader_for_file(self, file_path: str):
        """Get appropriate document loader for file type"""
        file_extension = Path(file_path).suffix.lower()
        
        if file_extension not in self.supported_extensions:
            raise ValueError(f"Unsupported file type: {file_extension}")
        
        loader_class = self.supported_extensions[file_extension]
        return loader_class(file_path)
    
    def _extract_metadata(self, file_path: str, additional_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Extract metadata from file"""
        file_info = Path(file_path)
        
        metadata = {
            "file_name": file_info.name,
            "file_path": str(file_path),
            "file_size": file_info.stat().st_size,
            "file_extension": file_info.suffix.lower(),
            "created_at": file_info.stat().st_ctime,
            "modified_at": file_info.stat().st_mtime,
            "ingestion_id": str(uuid.uuid4()),
            "ingestion_timestamp": "now"  # Will be replaced with actual timestamp
        }
        
        if additional_metadata:
            metadata.update(additional_metadata)
        
        return metadata
    
    async def ingest_file(self, 
                         file_path: str, 
                         collection_name: str = "documents",
                         namespace: str = "default",
                         additional_metadata: Optional[Dict[str, Any]] = None) -> List[str]:
        """Ingest a single file into the vector store"""
        try:
            # Validate file
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            if file_size_mb > settings.max_doc_size_mb:
                raise ValueError(f"File too large: {file_size_mb:.2f}MB > {settings.max_doc_size_mb}MB")
            
            # Load document
            loader = self._get_loader_for_file(file_path)
            documents = loader.load()
            
            # Extract metadata
            base_metadata = self._extract_metadata(file_path, additional_metadata)
            
            # Split documents into chunks
            chunks = []
            for doc in documents:
                # Update document metadata
                doc.metadata.update(base_metadata)
                doc.metadata["source_page"] = doc.metadata.get("page", 0)
                
                # Split document into chunks
                split_docs = self.text_splitter.split_documents([doc])
                
                # Add chunk-specific metadata
                for i, chunk in enumerate(split_docs):
                    chunk.metadata["chunk_id"] = i
                    chunk.metadata["total_chunks"] = len(split_docs)
                    chunk.metadata["chunk_size"] = len(chunk.page_content)
                    chunks.append(chunk)
            
            # Get vector store for collection
            vector_store = get_vector_store(collection_name)
            
            # Add documents to vector store
            doc_ids = await vector_store.add_documents(
                documents=chunks,
                namespace=namespace
            )
            
            # Store ingestion record in shared memory
            memory_store = get_shared_memory_store()
            ingestion_record = {
                "file_path": file_path,
                "collection_name": collection_name,
                "namespace": namespace,
                "document_ids": doc_ids,
                "chunk_count": len(chunks),
                "total_characters": sum(len(chunk.page_content) for chunk in chunks),
                "metadata": base_metadata
            }
            
            ingestion_namespace = MemoryNamespaces.SYSTEM_CONFIG + ("ingestion_records",)
            await memory_store.put_memory(
                namespace=ingestion_namespace,
                key=base_metadata["ingestion_id"],
                value=ingestion_record
            )
            
            logger.info(f"Successfully ingested {file_path}: {len(chunks)} chunks, {len(doc_ids)} documents")
            return doc_ids
            
        except Exception as e:
            logger.error(f"Failed to ingest file {file_path}: {e}")
            raise
    
    async def ingest_directory(self, 
                              directory_path: str,
                              collection_name: str = "documents",
                              namespace: str = "default",
                              recursive: bool = True,
                              additional_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, List[str]]:
        """Ingest all supported files from a directory"""
        try:
            directory = Path(directory_path)
            if not directory.exists() or not directory.is_dir():
                raise ValueError(f"Invalid directory: {directory_path}")
            
            # Find all supported files
            pattern = "**/*" if recursive else "*"
            all_files = list(directory.glob(pattern))
            
            supported_files = [
                f for f in all_files 
                if f.is_file() and f.suffix.lower() in self.supported_extensions
            ]
            
            if not supported_files:
                logger.warning(f"No supported files found in {directory_path}")
                return {}
            
            # Ingest files in batches
            results = {}
            batch_size = settings.ingestion_batch_size
            
            for i in range(0, len(supported_files), batch_size):
                batch = supported_files[i:i + batch_size]
                
                for file_path in batch:
                    try:
                        # Add directory-specific metadata
                        file_metadata = {"source_directory": str(directory_path)}
                        if additional_metadata:
                            file_metadata.update(additional_metadata)
                        
                        doc_ids = await self.ingest_file(
                            str(file_path),
                            collection_name=collection_name,
                            namespace=namespace,
                            additional_metadata=file_metadata
                        )
                        
                        results[str(file_path)] = doc_ids
                        
                    except Exception as e:
                        logger.error(f"Failed to ingest {file_path}: {e}")
                        results[str(file_path)] = []
            
            total_docs = sum(len(ids) for ids in results.values())
            logger.info(f"Directory ingestion complete: {len(supported_files)} files, {total_docs} documents")
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to ingest directory {directory_path}: {e}")
            raise
    
    async def ingest_text(self, 
                         text: str,
                         collection_name: str = "documents",
                         namespace: str = "default",
                         metadata: Optional[Dict[str, Any]] = None,
                         title: Optional[str] = None) -> List[str]:
        """Ingest raw text content"""
        try:
            # Create document from text
            doc_metadata = {
                "source": "text_input",
                "title": title or "Untitled Document",
                "content_type": "text",
                "ingestion_id": str(uuid.uuid4()),
                "character_count": len(text)
            }
            
            if metadata:
                doc_metadata.update(metadata)
            
            document = Document(page_content=text, metadata=doc_metadata)
            
            # Split into chunks
            chunks = self.text_splitter.split_documents([document])
            
            # Add chunk metadata
            for i, chunk in enumerate(chunks):
                chunk.metadata["chunk_id"] = i
                chunk.metadata["total_chunks"] = len(chunks)
                chunk.metadata["chunk_size"] = len(chunk.page_content)
            
            # Get vector store and add documents
            vector_store = get_vector_store(collection_name)
            doc_ids = await vector_store.add_documents(
                documents=chunks,
                namespace=namespace
            )
            
            logger.info(f"Successfully ingested text: {len(chunks)} chunks, {len(doc_ids)} documents")
            return doc_ids
            
        except Exception as e:
            logger.error(f"Failed to ingest text: {e}")
            raise
    
    async def delete_ingested_documents(self, ingestion_id: str) -> bool:
        """Delete all documents from a specific ingestion"""
        try:
            # Get ingestion record from memory
            memory_store = get_shared_memory_store()
            ingestion_namespace = MemoryNamespaces.SYSTEM_CONFIG + ("ingestion_records",)
            
            ingestion_record = await memory_store.get_memory(
                namespace=ingestion_namespace,
                key=ingestion_id
            )
            
            if not ingestion_record:
                logger.warning(f"Ingestion record not found: {ingestion_id}")
                return False
            
            # Delete documents from vector store
            collection_name = ingestion_record["collection_name"]
            document_ids = ingestion_record["document_ids"]
            
            vector_store = get_vector_store(collection_name)
            success = await vector_store.delete_documents(document_ids)
            
            if success:
                # Delete ingestion record
                await memory_store.delete_memory(
                    namespace=ingestion_namespace,
                    key=ingestion_id
                )
                
                logger.info(f"Successfully deleted ingestion {ingestion_id}: {len(document_ids)} documents")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to delete ingestion {ingestion_id}: {e}")
            return False
    
    async def get_ingestion_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get history of document ingestions"""
        try:
            memory_store = get_shared_memory_store()
            ingestion_namespace = MemoryNamespaces.SYSTEM_CONFIG + ("ingestion_records",)
            
            records = await memory_store.list_memories(
                namespace=ingestion_namespace,
                limit=limit
            )
            
            return [record["value"] for record in records]
            
        except Exception as e:
            logger.error(f"Failed to get ingestion history: {e}")
            return []

# Global document ingestion instance
document_ingestion = DocumentIngestion()

def get_document_ingestion() -> DocumentIngestion:
    """Get the global document ingestion instance"""
    return document_ingestion