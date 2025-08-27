# Technology Stack Rules for Ashworth Engine

## Overview
Rules and guidelines for technology stack usage in the Ashworth Engine that should be stored in the RAG system for dynamic retrieval by agents involved in development and implementation.

## Package Manager Preferences

### Primary Package Manager

1. **Yarn as Default**
   - Use yarn as the preferred package manager instead of pnpm
   - All commands involving package installation or execution must use yarn syntax
   - Document yarn usage in all setup and installation instructions

### Command Standards

1. **Installation Commands**
   - Use `yarn add` for adding dependencies
   - Use `yarn install` for installing all dependencies
   - Use `yarn dlx` for executing packages without installing

2. **Script Execution**
   - Define scripts in package.json using yarn conventions
   - Use `yarn run` for executing custom scripts
   - Maintain consistency across all package.json files

## Library Usage Policies

### Charting Libraries

1. **Primary Library**
   - Use Apache ECharts (via pyecharts) as the standard library for generating financial charts and visualizations
   - Do not use matplotlib directly for new chart implementations
   - Ensure all financial chart implementations use pyecharts

2. **Configuration Standards**
   - Configure pyecharts with business-appropriate themes and color schemes
   - Implement professional styling suitable for executive presentations
   - Support both web display and PNG export using snapshot-selenium

### Data Visualization

1. **Consistency Requirements**
   - Maintain consistent styling between web and print formats
   - Implement currency formatting for all financial data
   - Include interactive features for web display

2. **Export Capabilities**
   - Support PNG export for PDF report generation
   - Ensure high-quality images for professional presentations
   - Optimize file sizes for web delivery

## LLM Selection Strategy

### Model Selection

1. **Primary Models**
   - Use gpt-4o or gpt-4o-mini based on task complexity
   - Reserve gpt-3.5-turbo as fallback option
   - Verify implementation patterns against official LangGraph documentation

2. **Selection Criteria**
   - Choose based on task complexity and required reasoning
   - Consider cost and performance trade-offs
   - Ensure consistency within workflows

### Implementation Verification

1. **Pattern Compliance**
   - Follow official LangGraph documentation for implementation
   - Verify patterns against reference implementations
   - Test with multiple model variants

## Database Integration Standards

### Supabase Integration

1. **Connection Management**
   - Use local Supabase instance (127.0.0.1) for development and testing
   - Configure environment variables SUPABASE_URL and SUPABASE_KEY
   - Deploy using npx supabase secrets set with decryption keys

2. **Vector Database**
   - Implement using the pgvector extension in Supabase PostgreSQL
   - Use the vecs Python client for creating and managing vector collections
   - Configure with 1536-dimensional embeddings for RAG capabilities

### Data Modeling

1. **Schema Design**
   - Follow normalization principles
   - Implement proper indexing strategies
   - Include audit trails for compliance

2. **Security Practices**
   - Use parameterized queries to prevent injection
   - Implement proper authentication and authorization
   - Encrypt sensitive data at rest and in transit

## Development Toolchain

### Python Environment Management

1. **uv as Primary Tool**
   - Use uv for Python environment management
   - Leverage uv workspaces for monorepo projects
   - Ensure consistent dependency resolution

2. **Dependency Management**
   - Define dependencies in pyproject.toml
   - Use workspace dependencies for internal packages
   - Maintain uv.lock for reproducible builds

### Containerization

1. **Docker Standards**
   - Use Docker Compose for orchestration
   - Implement multi-stage builds for optimization
   - Define custom networks for service communication

2. **Environment Configuration**
   - Use .env files for configuration management
   - Implement consistent environment variable naming
   - Secure sensitive configuration data

## Implementation Guidance

### For Development Agents

1. **Tool Selection**
   - Always check RAG for approved tools and libraries
   - Follow established patterns and practices
   - Verify compatibility with existing stack

2. **Code Generation**
   - Use appropriate libraries and frameworks
   - Follow configuration standards
   - Include proper error handling

### For Configuration Agents

1. **Environment Setup**
   - Configure tools according to established standards
   - Document all configuration decisions
   - Ensure security and compliance

2. **Dependency Management**
   - Use approved package managers
   - Follow dependency resolution best practices
   - Maintain reproducible environments

## Examples

### Correct Package Manager Usage
```bash
# Installing dependencies
yarn add langchain

# Executing packages
yarn dlx create-turbo@latest

# Running scripts
yarn dev
```

### Incorrect Package Manager Usage
```bash
# Using pnpm instead of yarn
pnpm add langchain

# Using npx instead of yarn dlx
npx create-turbo@latest
```

## References

- Yarn documentation
- pyecharts documentation
- Supabase documentation
- LangGraph documentation
- uv documentation
- Docker Compose documentation