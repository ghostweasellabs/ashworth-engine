# Ashworth Engine v2 - Phase Plans Index

This document serves as the master index for all phase implementation plans for the Ashworth Engine v2 project.

## Project Overview

Ashworth Engine v2 is a full-stack, containerized financial intelligence platform featuring:
- A sophisticated multi-agent system built with Python and LangGraph
- A Retrieval-Augmented Generation (RAG) component using Supabase PostgreSQL with pgvector
- A responsive web application built with Next.js and shadcn/ui
- A polyglot monorepo managed by Turborepo and uv
- Fully containerized infrastructure using Docker Compose

## Implementation Phases

1. [Phase 1: Foundation Setup](phase-1-foundation-setup.md) - Establishing the monorepo foundation with Turborepo and uv
2. [Phase 2: Core Development](phase-2-core-development.md) - Containerizing the full stack with Docker and Supabase
3. [Phase 3: Advanced Analytics](phase-3-advanced-analytics.md) - Building the RAG knowledge ingestion pipeline
4. [Phase 4: IRS Compliance Enhancement](phase-4-irs-compliance.md) - Implementing tax categorization with official IRS guidelines
5. [Phase 5: Financial Visualization](phase-5-financial-visualization.md) - Adding chart generation capabilities
6. [Phase 6: Testing and Validation](phase-6-testing-validation.md) - Comprehensive testing and compliance verification
7. [Phase 7: Launch and Enhancements](phase-7-launch-enhancements.md) - Constructing the interactive frontend
8. [Phase 8: Testing and Containerization](phase-8-testing-containerization.md) - Implementing the LangGraph multi-agent system

## Supporting Documentation

- [RAG-Based Rule Management](rag-based-rule-management.md) - Storing project rules in the RAG system for dynamic retrieval
- [IRS Compliance Rules](irs-compliance-rules.md) - Official IRS compliance rules for tax processing
- [Development Process Rules](development-process-rules.md) - Development workflow and process standards
- [Agent Implementation Rules](agent-implementation-rules.md) - Guidelines for implementing agents
- [Technology Stack Rules](technology-stack-rules.md) - Technology stack usage policies and standards

## Getting Started

To implement the Ashworth Engine v2, follow the phases in sequential order. Each phase builds upon the previous one and includes specific checkpoints to verify successful completion.

## Technology Stack

- **Frontend**: Next.js, Turbopack, React, shadcn/ui
- **Backend**: Python, FastAPI, LangGraph
- **Database**: Supabase PostgreSQL with pgvector
- **Infrastructure**: Docker Compose, Turborepo, uv
- **AI/ML**: LangChain, Retrieval-Augmented Generation (RAG)

## Development Approach

This incremental build plan emphasizes:
- Build-and-test methodology
- Discrete, verifiable phases
- Solo developer optimization
- Containerized development environment
- Production-ready architecture