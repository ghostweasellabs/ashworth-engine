# ADR-007: PostgreSQL with pgvector for Analytics

## Status
Accepted

## Context
The Ashworth Engine requires both traditional relational data storage for financial transactions and vector search capabilities for RAG (Retrieval Augmented Generation) features. We need a solution that can handle both use cases efficiently.

## Decision
We will use PostgreSQL with the pgvector extension for both relational analytics and vector operations, accessed through Supabase.

## Rationale
- **Unified Database**: Single database for both relational and vector data
- **Proven Performance**: PostgreSQL is battle-tested for analytical workloads
- **Vector Search**: pgvector provides efficient similarity search for embeddings
- **ACID Compliance**: Ensures data consistency for financial data
- **Rich Query Capabilities**: Complex analytical queries with window functions, CTEs
- **RAG Integration**: Can query across analyses and provide contextual recommendations

## Consequences
### Positive
- Eliminates need for separate vector database (Pinecone, Weaviate)
- Complex cross-table analytics queries
- Consistent backup and recovery for all data types
- ACID guarantees for financial data integrity
- Rich indexing options for both relational and vector data

### Negative
- Single point of failure for both data types
- May not scale as efficiently as specialized vector databases
- Requires careful tuning for mixed workload performance
- pgvector extension dependency

## Implementation
- Use Supabase PostgreSQL instance with pgvector extension
- Create separate tables for relational data (transactions, clients, analyses)
- Create `documents` table with vector column for embeddings
- Use `vecs` Python client for vector operations
- Implement HNSW indexes for efficient vector similarity search
- Configure connection pooling for performance

## Data Architecture
- **Relational Tables**: clients, analyses, transactions, reports
- **Vector Table**: documents with 1536-dimensional embeddings
- **Cross-Query Capabilities**: JOIN operations between relational and vector data
- **Analytics Views**: Materialized views for common analytical queries

## Performance Considerations
- Use appropriate indexes (B-tree for relational, HNSW for vectors)
- Configure `work_mem` and `shared_buffers` for analytical workloads
- Monitor query performance and optimize as needed
- Consider read replicas for heavy analytical queries
- Chart generation uses pyecharts (Apache ECharts) for high-quality visualizations