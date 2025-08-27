# Phase 1: Foundation Setup - Establishing the Monorepo Foundation

## Objective

Construct the project's skeleton: a hybrid monorepo that elegantly accommodates both JavaScript/TypeScript and Python ecosystems. This phase establishes the foundational structure and tooling configuration.

## Key Technologies

- Turborepo for high-level task orchestration
- uv for Python environment management
- Next.js for frontend application
- FastAPI for backend application

## Implementation Steps

### 1.1 Workspace Architecture and Initial Scaffolding

1. Bootstrap the foundation using Turborepo's CLI:
   ```bash
   yarn dlx create-turbo@latest
   ```

2. Customize the generated structure to support mixed-language environment with the following canonical directory structure:

   ```
   apps/
     web/           # Next.js frontend application
     api/           # Python backend with FastAPI and LangGraph
   
   packages/
     ui/            # Shared React components with shadcn/ui
     eslint-config-custom/  # Centralized ESLint configuration
     tsconfig/      # Centralized TypeScript configuration
     py-common/     # Shared Python utilities and data models
   
   infra/
     docker-compose.yml  # Container orchestration
     supabase/      # Supabase CLI configuration
   ```

### 1.2 Configuring Turborepo for a Polyglot Environment

1. Define the primary Turborepo configuration in `turbo.json` at the repository root:

   ```json
   {
     "$schema": "https://turborepo.org/schema.json",
     "pipeline": {
       "build": {
         "dependsOn": ["^build"],
         "outputs": [".next/**", "!.next/cache/**", "dist/**"]
       },
       "lint": {},
       "dev": {
         "cache": false,
         "persistent": true
       },
       "clean": {
         "cache": false
       }
     }
   }
   ```

2. Configure scripts in root `package.json` to delegate execution to Turborepo:
   - `dev` script to concurrently launch Next.js and FastAPI development servers

### 1.3 Python Workspace Integration with uv

1. Create a `pyproject.toml` file at the monorepo root to define the uv workspace:
   ```toml
   [tool.uv.workspace]
   members = ["apps/api", "packages/py-common"]
   ```

2. Configure the `apps/api/pyproject.toml` to declare dependency on local py-common package:
   ```toml
   [project]
   name = "api"
   dependencies = [
       "py-common"
   ]
   
   [tool.uv.sources]
   py-common = { workspace = true }
   ```

### 1.4 Monorepo Directory and Configuration Overview

| Path | Purpose | Key Technologies |
| :---- | :---- | :---- |
| apps/web/ | Next.js frontend application | Next.js, Turbopack, React |
| apps/api/ | Python backend application | FastAPI, LangGraph, uv |
| packages/ui/ | Shared shadcn/ui components | React, Tailwind CSS |
| packages/py-common/ | Shared Python code (e.g., data models) | Python, uv |
| infra/ | Docker and Supabase configurations | Docker Compose |
| turbo.json | Monorepo task orchestration | Turborepo |
| pyproject.toml (root) | Defines the Python workspace | uv |

## Checkpoint 1

At the conclusion of this phase, the monorepo foundation should be solid:
- Execute `yarn install` from the root to install all JavaScript dependencies
- Create the unified Python virtual environment managed by uv
- Running `yarn dev` should successfully launch:
  - Next.js frontend development server
  - Placeholder FastAPI backend server
- The Python environment should be fully self-contained and reproducible

## Success Criteria

- [ ] Turborepo workspace initialized and configured
- [ ] uv workspace configured for Python packages
- [ ] Directory structure matches the canonical layout
- [ ] Root package.json contains proper scripts for development
- [ ] Both JavaScript and Python dependencies can be installed
- [ ] Development servers can be launched concurrently