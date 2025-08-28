-- Initial schema migration for Ashworth Engine
-- This migration sets up the core database structure

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "vector";

-- Create workflow checkpoints table for LangGraph state persistence
CREATE TABLE workflow_checkpoints (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    workflow_id VARCHAR(255) NOT NULL,
    checkpoint_id VARCHAR(255) NOT NULL,
    state JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    CONSTRAINT unique_workflow_checkpoint UNIQUE(workflow_id, checkpoint_id)
);

-- Create agent memory table for persistent agent state
CREATE TABLE agent_memory (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_id VARCHAR(255) NOT NULL,
    memory_key VARCHAR(255) NOT NULL,
    memory_value JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    CONSTRAINT unique_agent_memory UNIQUE(agent_id, memory_key)
);

-- Create document embeddings table for RAG system
CREATE TABLE document_embeddings (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id VARCHAR(255) NOT NULL,
    chunk_id VARCHAR(255) NOT NULL,
    content TEXT NOT NULL,
    embedding VECTOR(1536), -- OpenAI ada-002 embedding dimension
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    CONSTRAINT unique_document_chunk UNIQUE(document_id, chunk_id)
);

-- Create workflow runs table for tracking execution
CREATE TABLE workflow_runs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    workflow_id VARCHAR(255) NOT NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'pending',
    input_data JSONB DEFAULT '{}',
    output_data JSONB DEFAULT '{}',
    error_message TEXT,
    started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE,
    duration_ms INTEGER,
    
    CONSTRAINT valid_status CHECK (status IN ('pending', 'running', 'completed', 'failed', 'cancelled'))
);

-- Create indexes for performance
CREATE INDEX idx_workflow_checkpoints_workflow_id ON workflow_checkpoints(workflow_id);
CREATE INDEX idx_workflow_checkpoints_created_at ON workflow_checkpoints(created_at DESC);
CREATE INDEX idx_agent_memory_agent_id ON agent_memory(agent_id);
CREATE INDEX idx_agent_memory_created_at ON agent_memory(created_at DESC);
CREATE INDEX idx_document_embeddings_document_id ON document_embeddings(document_id);
CREATE INDEX idx_workflow_runs_workflow_id ON workflow_runs(workflow_id);
CREATE INDEX idx_workflow_runs_status ON workflow_runs(status);
CREATE INDEX idx_workflow_runs_started_at ON workflow_runs(started_at DESC);

-- Create vector similarity search index (will be created after data is inserted)
-- CREATE INDEX idx_document_embeddings_vector 
-- ON document_embeddings USING ivfflat (embedding vector_cosine_ops) 
-- WITH (lists = 100);

-- Create RLS policies (Row Level Security)
ALTER TABLE workflow_checkpoints ENABLE ROW LEVEL SECURITY;
ALTER TABLE agent_memory ENABLE ROW LEVEL SECURITY;
ALTER TABLE document_embeddings ENABLE ROW LEVEL SECURITY;
ALTER TABLE workflow_runs ENABLE ROW LEVEL SECURITY;

-- Create policies for authenticated users (adjust as needed for your auth setup)
CREATE POLICY "Allow all operations for authenticated users" ON workflow_checkpoints
    FOR ALL USING (true);

CREATE POLICY "Allow all operations for authenticated users" ON agent_memory
    FOR ALL USING (true);

CREATE POLICY "Allow all operations for authenticated users" ON document_embeddings
    FOR ALL USING (true);

CREATE POLICY "Allow all operations for authenticated users" ON workflow_runs
    FOR ALL USING (true);

-- Create updated_at trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers for updated_at columns
CREATE TRIGGER update_workflow_checkpoints_updated_at 
    BEFORE UPDATE ON workflow_checkpoints 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_agent_memory_updated_at 
    BEFORE UPDATE ON agent_memory 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();