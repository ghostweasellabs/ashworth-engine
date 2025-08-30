-- Fix document_embeddings table schema
-- Drop existing table if it exists and recreate with correct schema

DROP TABLE IF EXISTS document_embeddings CASCADE;

-- Create table for storing document embeddings with correct schema
CREATE TABLE document_embeddings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    embedding VECTOR(1536) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for efficient similarity search
CREATE INDEX document_embeddings_embedding_idx 
ON document_embeddings USING ivfflat (embedding vector_cosine_ops) 
WITH (lists = 100);

-- Create index for document_id lookups
CREATE INDEX document_embeddings_document_id_idx 
ON document_embeddings (document_id);

-- Create index for chunk_index lookups
CREATE INDEX document_embeddings_chunk_index_idx 
ON document_embeddings (chunk_index);

-- Create index for metadata queries
CREATE INDEX document_embeddings_metadata_idx 
ON document_embeddings USING gin (metadata);

-- Trigger to automatically update updated_at
CREATE TRIGGER update_document_embeddings_updated_at 
    BEFORE UPDATE ON document_embeddings 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();