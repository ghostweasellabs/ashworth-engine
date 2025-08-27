# Phase 2: Core Development - Containerizing the Full Stack with Docker and Supabase

## Objective

Transition the project from a locally-run setup to a fully containerized environment using Docker Compose. This ensures development parity with production, eliminates environment-specific issues, and simplifies setup.

## Key Technologies

- Docker Compose for container orchestration
- Supabase for data and authentication layer
- Environment variable management

## Implementation Steps

### 2.1 Local Supabase Setup with the CLI

1. Initialize Supabase within the `infra/` directory:
   ```bash
   supabase init
   ```

2. Start Supabase services:
   ```bash
   supabase start
   ```

3. Verify successful startup by accessing Supabase Studio at http://localhost:54323

### 2.2 Orchestration with a Unified docker-compose.yml

1. Create a root `docker-compose.yml` file in the `infra/` directory

2. Use Docker Compose's include feature to integrate Supabase services:
   ```yaml
   services:
     web:
       build:
         context: ../
         dockerfile: ./apps/web/Dockerfile
       ports:
         - "3000:3000"
       env_file:
         - .env
       networks:
         - ashworth-net
   
     api:
       build:
         context: ../
         dockerfile: ./apps/api/Dockerfile
       ports:
         - "8001:8000" # Expose API on 8001 to avoid conflict with Supabase's Kong
       env_file:
         - .env
       networks:
         - ashworth-net
   
   # Include Supabase services and attach them to our network
   include:
     - path: ./supabase/docker-compose.yml
       env_file: .env
   
   networks:
     ashworth-net:
       driver: bridge
   ```

3. Define a custom Docker network for inter-service communication

### 2.3 Unified Environment Management

1. Create a single `.env` file at the root of the `infra/` directory as the single source of truth for all configuration variables

2. Use the `env_file` directive in `docker-compose.yml` to inject variables into appropriate containers

### 2.4 Service Environment Variables

Configure the following essential environment variables:

| Variable | Service(s) | Example Value | Purpose |
| :---- | :---- | :---- | :---- |
| POSTGRES_PASSWORD | Supabase | your-super-secret-password | Sets the local Postgres superuser password |
| SUPABASE_ANON_KEY | Supabase, Web | ey... | Public-facing key for client-side Supabase access |
| SUPABASE_SERVICE_ROLE_KEY | Supabase, API | ey... | Secret key for backend services to bypass RLS |
| DATABASE_URL | API | postgresql://postgres:...@db:5432/postgres | Direct connection string for the API to the Postgres container over the Docker network |
| NEXT_PUBLIC_SUPABASE_URL | Web | http://localhost:8000 | URL for the browser to reach the local Supabase API gateway (Kong) |
| NEXT_PUBLIC_API_URL | Web | http://localhost:8001 | URL for the browser to reach the local FastAPI backend |
| INTERNAL_SUPABASE_URL | API | http://kong:8000 | URL for the API container to reach the Supabase gateway over the Docker network |

## Checkpoint 2

With the completion of this phase, the entire platform should be containerized:
- Execute `docker compose up --build` from the `infra/` directory
- Build application images and launch the entire stack:
  - Next.js frontend
  - FastAPI backend
  - All local Supabase services
- Services should be healthy and capable of communicating over the custom Docker network
- Access the web application at http://localhost:3000
- Access the local Supabase Studio at http://localhost:54323

## Success Criteria

- [ ] Supabase local instance running and accessible via Studio
- [ ] docker-compose.yml properly configured with all services
- [ ] Custom Docker network enables inter-service communication
- [ ] Environment variables properly configured for all services
- [ ] All services start successfully with `docker compose up --build`
- [ ] Web application accessible at http://localhost:3000
- [ ] Supabase Studio accessible at http://localhost:54323