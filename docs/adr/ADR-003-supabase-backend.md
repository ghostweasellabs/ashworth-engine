# ADR-003: Supabase as Backend Infrastructure

## Status
Accepted

## Context
The Ashworth Engine requires comprehensive backend infrastructure including PostgreSQL database, vector search capabilities, file storage, authentication, and real-time features. Building these from scratch would be time-consuming and error-prone.

## Decision
We will use Supabase as our backend-as-a-service platform, providing PostgreSQL with pgvector extension, authentication, storage, and real-time capabilities.

## Rationale
- **Comprehensive Solution**: Provides database, auth, storage, and real-time in one platform
- **PostgreSQL + pgvector**: Native vector database support for RAG capabilities
- **Local Development**: Full local stack with `supabase start` for development
- **Modern Architecture**: REST APIs, real-time subscriptions, and edge functions
- **Open Source**: Built on open-source technologies (PostgreSQL, PostgREST)

## Consequences
### Positive
- Eliminates need for separate MinIO, Redis, or custom auth solutions
- Built-in vector search capabilities with pgvector
- Excellent local development experience
- Automatic API generation from database schema
- Built-in Row Level Security (RLS) for data protection

### Negative
- External service dependency for production
- Vendor lock-in to Supabase ecosystem
- Learning curve for Supabase-specific features
- Potential costs for high-volume usage

## Implementation
- Use local Supabase instance (127.0.0.1) for development and testing
- Configure `supabase/config.toml` for project settings
- Create database migrations in `supabase/migrations/`
- Use `vecs` Python client for vector operations
- Configure environment variables for local connections

## Configuration Requirements
- Local development: `SUPABASE_URL=http://127.0.0.1:54321`
- Database: `DATABASE_URL=postgresql://postgres:postgres@127.0.0.1:54322/postgres`
- Vector operations: Use vecs client with local PostgreSQL connection
- Storage buckets: Configure for reports and charts