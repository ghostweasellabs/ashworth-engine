# ADR-006: Containerize with Docker Compose

## Status
Accepted

## Context
The Ashworth Engine needs a deployment strategy that is simple for solo development while providing reproducible, local-first deployment options. The system requires coordination between multiple services (API, database, storage).

## Decision
We will deploy using Docker Compose, including the full Supabase stack, optimized for solo development and local-first deployment.

## Rationale
- **Simplicity**: Docker Compose provides straightforward multi-service orchestration
- **Solo Developer Friendly**: Minimal complexity for single-developer teams
- **Reproducible Environment**: Consistent deployments across development and production
- **Local-First**: Can run entirely offline once images are downloaded
- **Supabase Integration**: Supabase provides official Docker Compose setup

## Consequences
### Positive
- Easy local development setup with `docker-compose up`
- Reproducible deployments across environments
- Simple backup and migration strategies
- No complex Kubernetes overhead for small deployments
- Easy integration with CI/CD pipelines

### Negative
- Limited horizontal scalability compared to Kubernetes
- Single-machine deployment limitations
- Less sophisticated load balancing and auto-scaling
- Manual management of container health and updates

## Implementation
- Use `supabase start` for local development (uses Docker Compose internally)
- Create production `docker-compose.yml` for deployment
- Configure volume persistence for database and storage
- Set up proper networking between services
- Include monitoring and logging containers

## Service Architecture
- **API Service**: FastAPI application container
- **Database**: PostgreSQL with pgvector (via Supabase)
- **Storage**: Supabase storage service
- **Auth**: Supabase auth service  
- **Monitoring**: Optional Prometheus/Grafana containers

## Future Considerations
- Evaluate Kubernetes migration for high-scale deployments
- Consider managed container services (ECS, Cloud Run) for production
- Assess need for service mesh for microservices communication