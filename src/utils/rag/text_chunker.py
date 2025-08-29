"""
Text chunking utilities for document processing.
Handles intelligent chunking with overlap and metadata preservation.
"""

import logging
import re
from typing import Dict, List, Optional

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class TextChunk(BaseModel):
    """A chunk of text with metadata."""
    content: str
    chunk_index: int
    metadata: Dict
    start_char: int = 0
    end_char: int = 0
    embedding: Optional[List[float]] = None


class TextChunker:
    """Intelligent text chunking for document processing."""
    
    def __init__(
        self, 
        chunk_size: int = 1000, 
        chunk_overlap: int = 200,
        min_chunk_size: int = 100
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        
    def chunk_text(self, text: str, metadata: Dict = None) -> List[TextChunk]:
        """
        Chunk text into overlapping segments with intelligent splitting.
        
        Args:
            text: Text to chunk
            metadata: Base metadata to include with each chunk
            
        Returns:
            List of TextChunk objects
        """
        if metadata is None:
            metadata = {}
            
        # Clean the text
        text = self._clean_text(text)
        
        # Try semantic chunking first (by sections/paragraphs)
        chunks = self._semantic_chunk(text, metadata)
        
        # If semantic chunking didn't work well, fall back to sliding window
        if not chunks or len(chunks) == 1 and len(text) > self.chunk_size * 2:
            chunks = self._sliding_window_chunk(text, metadata)
        
        # Filter out chunks that are too small
        chunks = [chunk for chunk in chunks if len(chunk.content.strip()) >= self.min_chunk_size]
        
        logger.info(f"Created {len(chunks)} chunks from text of length {len(text)}")
        return chunks
    
    def _semantic_chunk(self, text: str, base_metadata: Dict) -> List[TextChunk]:
        """Chunk text based on semantic boundaries (sections, paragraphs)."""
        chunks = []
        
        # Split by major sections first (Chapter, Section, etc.)
        section_pattern = r'\n\s*(Chapter|Section|Part)\s+\d+[:\-\s]'
        sections = re.split(section_pattern, text, flags=re.IGNORECASE)
        
        if len(sections) > 1:
            # We found sections, process each one
            current_pos = 0
            for i, section in enumerate(sections):
                if not section.strip():
                    continue
                    
                section_chunks = self._chunk_section(section, base_metadata, i, current_pos)
                chunks.extend(section_chunks)
                current_pos += len(section)
        else:
            # No clear sections, try paragraph-based chunking
            chunks = self._paragraph_chunk(text, base_metadata)
        
        return chunks
    
    def _chunk_section(self, section: str, base_metadata: Dict, section_index: int, start_pos: int) -> List[TextChunk]:
        """Chunk a section of text."""
        chunks = []
        
        # Extract section title if present
        lines = section.split('\n')
        section_title = lines[0].strip() if lines else ""
        
        # If section is small enough, keep as one chunk
        if len(section) <= self.chunk_size:
            metadata = base_metadata.copy()
            metadata.update({
                'section_index': section_index,
                'section_title': section_title,
                'chunk_type': 'section'
            })
            
            chunk = TextChunk(
                content=section.strip(),
                chunk_index=len(chunks),
                metadata=metadata,
                start_char=start_pos,
                end_char=start_pos + len(section)
            )
            chunks.append(chunk)
        else:
            # Section is too large, split by paragraphs
            paragraphs = section.split('\n\n')
            current_chunk = ""
            current_start = start_pos
            
            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue
                
                # If adding this paragraph would exceed chunk size, finalize current chunk
                if current_chunk and len(current_chunk) + len(para) > self.chunk_size:
                    metadata = base_metadata.copy()
                    metadata.update({
                        'section_index': section_index,
                        'section_title': section_title,
                        'chunk_type': 'section_part'
                    })
                    
                    chunk = TextChunk(
                        content=current_chunk.strip(),
                        chunk_index=len(chunks),
                        metadata=metadata,
                        start_char=current_start,
                        end_char=current_start + len(current_chunk)
                    )
                    chunks.append(chunk)
                    
                    # Start new chunk with overlap
                    overlap_text = self._get_overlap_text(current_chunk)
                    current_chunk = overlap_text + "\n\n" + para
                    current_start = current_start + len(current_chunk) - len(overlap_text) - len(para) - 2
                else:
                    # Add paragraph to current chunk
                    if current_chunk:
                        current_chunk += "\n\n" + para
                    else:
                        current_chunk = para
            
            # Add final chunk if there's content
            if current_chunk.strip():
                metadata = base_metadata.copy()
                metadata.update({
                    'section_index': section_index,
                    'section_title': section_title,
                    'chunk_type': 'section_part'
                })
                
                chunk = TextChunk(
                    content=current_chunk.strip(),
                    chunk_index=len(chunks),
                    metadata=metadata,
                    start_char=current_start,
                    end_char=current_start + len(current_chunk)
                )
                chunks.append(chunk)
        
        return chunks
    
    def _paragraph_chunk(self, text: str, base_metadata: Dict) -> List[TextChunk]:
        """Chunk text by paragraphs when no clear sections are found."""
        chunks = []
        paragraphs = text.split('\n\n')
        
        current_chunk = ""
        current_start = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # If adding this paragraph would exceed chunk size, finalize current chunk
            if current_chunk and len(current_chunk) + len(para) > self.chunk_size:
                metadata = base_metadata.copy()
                metadata.update({'chunk_type': 'paragraph'})
                
                chunk = TextChunk(
                    content=current_chunk.strip(),
                    chunk_index=len(chunks),
                    metadata=metadata,
                    start_char=current_start,
                    end_char=current_start + len(current_chunk)
                )
                chunks.append(chunk)
                
                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_chunk)
                current_chunk = overlap_text + "\n\n" + para
                current_start = current_start + len(current_chunk) - len(overlap_text) - len(para) - 2
            else:
                # Add paragraph to current chunk
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
        
        # Add final chunk if there's content
        if current_chunk.strip():
            metadata = base_metadata.copy()
            metadata.update({'chunk_type': 'paragraph'})
            
            chunk = TextChunk(
                content=current_chunk.strip(),
                chunk_index=len(chunks),
                metadata=metadata,
                start_char=current_start,
                end_char=current_start + len(current_chunk)
            )
            chunks.append(chunk)
        
        return chunks
    
    def _sliding_window_chunk(self, text: str, base_metadata: Dict) -> List[TextChunk]:
        """Fallback chunking using sliding window approach."""
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            
            # Try to end at a sentence boundary
            if end < len(text):
                # Look for sentence endings within the last 100 characters
                search_start = max(end - 100, start)
                sentence_end = text.rfind('.', search_start, end)
                if sentence_end > start:
                    end = sentence_end + 1
            
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                metadata = base_metadata.copy()
                metadata.update({'chunk_type': 'sliding_window'})
                
                chunk = TextChunk(
                    content=chunk_text,
                    chunk_index=len(chunks),
                    metadata=metadata,
                    start_char=start,
                    end_char=end
                )
                chunks.append(chunk)
            
            # Move start position with overlap
            start = max(start + self.chunk_size - self.chunk_overlap, end)
        
        return chunks
    
    def _get_overlap_text(self, text: str) -> str:
        """Get overlap text from the end of a chunk."""
        if len(text) <= self.chunk_overlap:
            return text
        
        # Try to get overlap at sentence boundary
        overlap_start = len(text) - self.chunk_overlap
        sentence_start = text.find('.', overlap_start)
        
        if sentence_start > overlap_start:
            return text[sentence_start + 1:].strip()
        else:
            return text[-self.chunk_overlap:].strip()
    
    def _clean_text(self, text: str) -> str:
        """Clean text for better chunking."""
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove excessive line breaks but preserve paragraph structure
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        return text.strip()


# Test function
def test_chunking():
    """Test the text chunker with sample content."""
    chunker = TextChunker(chunk_size=500, chunk_overlap=100)
    
    sample_text = """
    Chapter 1: Business Income
    
    Generally, you must include in gross income everything you receive in payment for personal services. In addition to wages, salaries, commissions, fees, and tips, this includes other forms of compensation such as fringe benefits and stock options.
    
    Business income includes income from services you performed, income from the sale or use of products or property, and income from investments. You must report all business income on your tax return.
    
    Chapter 2: Business Expenses
    
    To be deductible, a business expense must be both ordinary and necessary. An ordinary expense is one that is common and accepted in your trade or business. A necessary expense is one that is helpful and appropriate for your trade or business.
    
    Common business expenses include office supplies, travel, meals, and professional services.
    """
    
    chunks = chunker.chunk_text(sample_text, {'document_type': 'test'})
    
    print(f"Created {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i + 1} ({len(chunk.content)} chars):")
        print(f"Type: {chunk.metadata.get('chunk_type', 'unknown')}")
        print(f"Content: {chunk.content[:100]}...")


if __name__ == "__main__":
    test_chunking()