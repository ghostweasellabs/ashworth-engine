-- Enable pgvector extension for vector similarity search
CREATE EXTENSION IF NOT EXISTS vector;

-- Create table for storing document embeddings
CREATE TABLE IF NOT EXISTS document_embeddings (
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
CREATE INDEX IF NOT EXISTS document_embeddings_embedding_idx 
ON document_embeddings USING ivfflat (embedding vector_cosine_ops) 
WITH (lists = 100);

-- Create index for document_id lookups
CREATE INDEX IF NOT EXISTS document_embeddings_document_id_idx 
ON document_embeddings (document_id);

-- Create index for metadata queries
CREATE INDEX IF NOT EXISTS document_embeddings_metadata_idx 
ON document_embeddings USING gin (metadata);

-- Create table for storing source documents
CREATE TABLE IF NOT EXISTS source_documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id TEXT UNIQUE NOT NULL,
    title TEXT NOT NULL,
    source_url TEXT,
    document_type TEXT NOT NULL DEFAULT 'irs_publication',
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    ingested_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create index for document lookups
CREATE INDEX IF NOT EXISTS source_documents_document_id_idx 
ON source_documents (document_id);

-- Create index for document type filtering
CREATE INDEX IF NOT EXISTS source_documents_type_idx 
ON source_documents (document_type);

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Triggers to automatically update updated_at
CREATE TRIGGER update_document_embeddings_updated_at 
    BEFORE UPDATE ON document_embeddings 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_source_documents_updated_at 
    BEFORE UPDATE ON source_documents 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();