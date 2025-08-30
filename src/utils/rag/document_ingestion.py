"""
Document ingestion pipeline for IRS publications and other rule documents.
Handles downloading, chunking, embedding, and storing documents in pgvector.
"""

import asyncio
import hashlib
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse

import aiohttp
from pydantic import BaseModel, Field

from src.config.settings import get_settings
from src.utils.rag.embeddings import EmbeddingGenerator
from src.utils.rag.text_chunker import TextChunker
from src.utils.rag.vector_store import VectorStore

logger = logging.getLogger(__name__)


class DocumentMetadata(BaseModel):
    """Metadata for ingested documents."""
    title: str
    source_url: Optional[str] = None
    document_type: str = "irs_publication"
    publication_year: Optional[int] = None
    section: Optional[str] = None
    category: Optional[str] = None
    keywords: List[str] = Field(default_factory=list)


class DocumentChunk(BaseModel):
    """A chunk of document content with metadata."""
    content: str
    chunk_index: int
    metadata: Dict
    embedding: Optional[List[float]] = None


class DocumentIngestionPipeline:
    """Pipeline for ingesting documents into the vector database."""
    
    def __init__(self):
        self.settings = get_settings()
        self.embedding_generator = EmbeddingGenerator()
        self.text_chunker = TextChunker()
        self.vector_store = VectorStore()
        
    async def ingest_document(
        self, 
        content: str, 
        document_id: str, 
        metadata: DocumentMetadata
    ) -> bool:
        """
        Ingest a document into the vector database.
        
        Args:
            content: The document content
            document_id: Unique identifier for the document
            metadata: Document metadata
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Starting ingestion for document: {document_id}")
            
            # Store source document
            await self.vector_store.store_source_document(
                document_id=document_id,
                title=metadata.title,
                source_url=metadata.source_url,
                document_type=metadata.document_type,
                content=content,
                metadata=metadata.dict()
            )
            
            # Chunk the document
            chunks = self.text_chunker.chunk_text(content, metadata.dict())
            logger.info(f"Created {len(chunks)} chunks for document {document_id}")
            
            # Generate embeddings for chunks
            chunk_embeddings = []
            for i, chunk in enumerate(chunks):
                try:
                    embedding = await self.embedding_generator.generate_embedding(chunk.content)
                    if embedding:
                        # Create a new chunk with the embedding
                        chunk_with_embedding = chunk.model_copy()
                        chunk_with_embedding.embedding = embedding
                        chunk_embeddings.append(chunk_with_embedding)
                    
                    if (i + 1) % 10 == 0:
                        logger.info(f"Generated embeddings for {i + 1}/{len(chunks)} chunks")
                        
                except Exception as e:
                    logger.error(f"Failed to generate embedding for chunk {i}: {e}")
                    continue
            
            # Store embeddings in vector database
            await self.vector_store.store_embeddings(document_id, chunk_embeddings)
            
            logger.info(f"Successfully ingested document {document_id} with {len(chunk_embeddings)} chunks")
            return True
            
        except Exception as e:
            logger.error(f"Failed to ingest document {document_id}: {e}")
            return False
    
    async def ingest_irs_publication(self, publication_url: str, publication_id: str) -> bool:
        """
        Download and ingest an IRS publication.
        
        Args:
            publication_url: URL to the IRS publication
            publication_id: Unique identifier (e.g., "pub334_2024")
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Downloading IRS publication from: {publication_url}")
            
            # Download the document
            content = await self._download_document(publication_url)
            if not content:
                logger.error(f"Failed to download document from {publication_url}")
                return False
            
            # Extract metadata from content
            metadata = self._extract_irs_metadata(content, publication_url)
            
            # Ingest the document
            return await self.ingest_document(content, publication_id, metadata)
            
        except Exception as e:
            logger.error(f"Failed to ingest IRS publication {publication_id}: {e}")
            return False
    
    async def _download_document(self, url: str) -> Optional[str]:
        """Download document content from URL."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        content = await response.text()
                        return content
                    else:
                        logger.error(f"HTTP {response.status} when downloading {url}")
                        return None
        except Exception as e:
            logger.error(f"Error downloading document from {url}: {e}")
            return None
    
    def _extract_irs_metadata(self, content: str, source_url: str) -> DocumentMetadata:
        """Extract metadata from IRS publication content."""
        # Extract title from content (usually in first few lines)
        lines = content.split('\n')[:20]
        title = "IRS Publication"
        
        for line in lines:
            line = line.strip()
            if line and len(line) > 10 and not line.startswith('Department'):
                # Look for publication title patterns
                if 'Publication' in line or 'Tax' in line or 'Business' in line:
                    title = line
                    break
        
        # Extract publication year
        year_match = re.search(r'20\d{2}', content[:1000])
        publication_year = int(year_match.group()) if year_match else None
        
        # Extract keywords from content
        keywords = self._extract_keywords(content)
        
        return DocumentMetadata(
            title=title,
            source_url=source_url,
            document_type="irs_publication",
            publication_year=publication_year,
            keywords=keywords
        )
    
    def _extract_keywords(self, content: str) -> List[str]:
        """Extract relevant keywords from document content."""
        # Common tax-related keywords to look for
        tax_keywords = [
            'deduction', 'expense', 'business', 'income', 'tax', 'irs',
            'depreciation', 'amortization', 'credit', 'liability',
            'schedule', 'form', 'return', 'filing', 'compliance'
        ]
        
        content_lower = content.lower()
        found_keywords = []
        
        for keyword in tax_keywords:
            if keyword in content_lower:
                found_keywords.append(keyword)
        
        return found_keywords
    
    async def ingest_sample_documents(self) -> bool:
        """Ingest sample IRS documents for testing."""
        try:
            # Sample IRS Publication 334 content (simplified for testing)
            sample_content = """
            Publication 334 - Tax Guide for Small Business
            
            Chapter 1: Business Income
            
            Generally, you must include in gross income everything you receive in payment for personal services. In addition to wages, salaries, commissions, fees, and tips, this includes other forms of compensation such as fringe benefits and stock options.
            
            Business income includes income from services you performed, income from the sale or use of products or property, and income from investments. You must report all business income on your tax return.
            
            Chapter 2: Business Expenses
            
            To be deductible, a business expense must be both ordinary and necessary. An ordinary expense is one that is common and accepted in your trade or business. A necessary expense is one that is helpful and appropriate for your trade or business.
            
            Common business expenses include:
            - Office supplies and equipment
            - Business meals (50% deductible)
            - Travel expenses
            - Professional services
            - Advertising and marketing
            - Insurance premiums
            - Rent or lease payments
            
            Chapter 3: Depreciation and Amortization
            
            Depreciation is the recovery of the cost of property over a number of years. You deduct a part of the cost every year until you fully recover its cost. Amortization is similar to depreciation but applies to intangible property.
            
            Section 179 allows you to deduct the full cost of qualifying property in the year you place it in service, subject to certain limits.
            """
            
            metadata = DocumentMetadata(
                title="Publication 334 - Tax Guide for Small Business",
                document_type="irs_publication",
                publication_year=2024,
                keywords=["business", "expense", "deduction", "depreciation", "income"]
            )
            
            success = await self.ingest_document(
                content=sample_content,
                document_id="pub334_sample_2024",
                metadata=metadata
            )
            
            if success:
                logger.info("Successfully ingested sample IRS document")
            else:
                logger.error("Failed to ingest sample IRS document")
                
            return success
            
        except Exception as e:
            logger.error(f"Error ingesting sample documents: {e}")
            return False
    
    async def get_ingestion_stats(self) -> Dict:
        """Get statistics about ingested documents."""
        try:
            stats = await self.vector_store.get_ingestion_stats()
            return stats
        except Exception as e:
            logger.error(f"Error getting ingestion stats: {e}")
            return {}


# Convenience function for quick ingestion
async def ingest_sample_irs_documents():
    """Quick function to ingest sample IRS documents."""
    pipeline = DocumentIngestionPipeline()
    return await pipeline.ingest_sample_documents()


if __name__ == "__main__":
    # Test the ingestion pipeline
    async def main():
        pipeline = DocumentIngestionPipeline()
        success = await pipeline.ingest_sample_documents()
        if success:
            stats = await pipeline.get_ingestion_stats()
            print(f"Ingestion complete. Stats: {stats}")
        else:
            print("Ingestion failed")
    
    asyncio.run(main())