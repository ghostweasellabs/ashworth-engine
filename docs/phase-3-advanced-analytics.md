# Phase 3: Advanced Analytics - Building the RAG Knowledge Ingestion Pipeline

## Objective

Implement the first core feature: a pipeline to ingest financial documents into the knowledge base. This involves setting up the vector database, creating a backend service to process documents, and storing them as embeddings for future retrieval.

## Key Technologies

- Supabase PostgreSQL with pgvector extension
- LangChain for document processing
- FastAPI for backend services
- Embedding models (OpenAI or Supabase/gte-small)

## Implementation Steps

### 3.1 Configuring the Supabase Vector Store

1. Enable pgvector extension using Supabase migrations:
   ```bash
   supabase migration new enable_pgvector
   ```
   Add to the migration file:
   ```sql
   create extension if not exists vector;
   ```

2. Create knowledge base table:
   ```bash
   supabase migration new create_documents_table
   ```
   Define the schema:
   ```sql
   create table documents (
     id bigserial primary key,
     content text,
     metadata jsonb,
     embedding vector(1536)  -- Match embedding model dimensions
   );
   ```

3. Create search function:
   ```bash
   supabase migration new create_match_documents_function
   ```
   Add the PL/pgSQL function for cosine similarity search

4. Apply migrations:
   ```bash
   supabase db reset
   ```

### 3.2 Developing the FastAPI Ingestion Service

1. Add a new endpoint `POST /upload` to the api service

2. Implement file handling using FastAPI's UploadFile:
   ```python
   from fastapi import FastAPI, UploadFile
   ```

3. Implement document loading with LangChain loaders:
   - Start with PyPDFLoader for PDF files
   - Design for extensibility to support multiple file types

4. Implement text splitting:
   - Use RecursiveCharacterTextSplitter
   - Configure chunk_size and chunk_overlap parameters

5. Implement embedding and storage:
   - Use LangChain SupabaseVectorStore
   - Implement from_documents() method for batch insertion
   - Connect to Supabase instance with provided client

## Checkpoint 3

The ingestion pipeline should be complete and testable:
- Use an API client (Insomnia/Postman) to send a POST request with a file to http://localhost:8001/upload
- After successful request, connect to the local PostgreSQL database
- Query the documents table to verify it has been populated with:
  - Multiple rows for each document
  - Text chunks in the content column
  - Vector embeddings in the embedding column

## Success Criteria

- [ ] pgvector extension enabled in Supabase
- [ ] Documents table created with proper schema
- [ ] Match documents function implemented for similarity search
- [ ] All migrations applied successfully
- [ ] POST /upload endpoint implemented in FastAPI
- [ ] Document loading working with PyPDFLoader
- [ ] Text splitting implemented with appropriate parameters
- [ ] Embedding generation and storage working
- [ ] End-to-end testing successful with API client