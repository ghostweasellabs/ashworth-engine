# Ashworth Engine v2 API Documentation

## Overview

The **Ashworth Engine v2** is a pure backend API for AI-powered financial intelligence with RAG (Retrieval-Augmented Generation) document management capabilities. This FastAPI-based service provides comprehensive financial analysis, document ingestion, and semantic search functionality.

## 🚀 Quick Start

### Starting the Server
```bash
# Start the development server
uv run --env-file .env python server.py

# Server will be available at:
# http://localhost:8000
```

### Interactive Documentation
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI Schema**: http://localhost:8000/openapi.json

## 📋 API Endpoints

### Financial Analysis
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/reports` | Create a new financial analysis report |
| `GET` | `/reports/{report_id}` | Get report status and metadata |
| `GET` | `/clients/{client_id}/reports` | Get reports for a specific client |

### RAG Document Management
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/rag/documents/upload` | Upload and ingest documents |
| `POST` | `/rag/documents/upload-text` | Upload raw text content |
| `POST` | `/rag/search` | Search documents with semantic similarity |
| `POST` | `/rag/setup-irs-knowledge` | Initialize IRS knowledge base |
| `GET` | `/rag/collections` | Get available collections and configuration |

### System
| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check with connectivity status |

## 🔧 API Usage Examples

### 1. Upload a Document
```bash
curl -X POST "http://localhost:8000/rag/documents/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@document.pdf" \
  -F "collection_name=user_documents" \
  -F "namespace=default" \
  -F "client_id=client_123" \
  -F "document_type=financial_statement"
```

### 2. Search Documents
```bash
curl -X POST "http://localhost:8000/rag/search" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "query=business expense deductions" \
  -d "collection_name=user_documents" \
  -d "top_k=5" \
  -d "score_threshold=0.8"
```

### 3. Create Financial Report
```bash
curl -X POST "http://localhost:8000/reports" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@transactions.csv" \
  -F "client_id=client_123" \
  -F "analysis_type=financial_analysis"
```

### 4. Health Check
```bash
curl http://localhost:8000/health
```

## 📊 Data Models

### ReportSummary
```json
{
  "report_id": "string",
  "status": "string",
  "summary": {},
  "warnings": ["string"],
  "report_url": "string",
  "error_message": "string",
  "storage_path": "string"
}
```

### DocumentUploadResponse
```json
{
  "document_id": "string",
  "status": "string",
  "filename": "string",
  "collection_name": "string",
  "namespace": "string",
  "chunks_created": 0,
  "ingestion_timestamp": "string",
  "file_size_bytes": 0,
  "metadata": {},
  "error_message": "string"
}
```

### DocumentSearchResponse
```json
{
  "query": "string",
  "results": [{}],
  "collection_name": "string",
  "namespace": "string",
  "search_timestamp": "string"
}
```

## 🗂️ RAG Collections

The system organizes documents into different collections:

- **`irs_documents`**: Official IRS publications and forms
- **`financial_regulations`**: Financial regulatory documents  
- **`tax_guidance`**: Tax optimization strategies and guidance
- **`user_documents`**: User-uploaded documents (default)

## 📁 Supported File Types

- PDF (`.pdf`)
- Text files (`.txt`)
- Markdown (`.md`)
- Word documents (`.docx`)
- CSV files (`.csv`)
- Excel files (`.xlsx`, `.xls`)

## ⚙️ Configuration

### Environment Variables
```bash
# LLM Configuration
OPENAI_API_KEY=your-key-here
LLM_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-ada-002

# Database Configuration
DATABASE_URL=postgresql://user:pass@localhost:5432/db
VECS_CONNECTION_STRING=postgresql://user:pass@localhost:5432/db

# Supabase Configuration
SUPABASE_URL=http://127.0.0.1:54321
SUPABASE_SERVICE_KEY=your-key-here

# Processing Configuration
MAX_UPLOAD_SIZE=52428800  # 50MB
VECTOR_DIMENSION=1536
VECTOR_SIMILARITY_THRESHOLD=0.8
```

### System Requirements
- Python 3.11+
- PostgreSQL with pgvector extension
- Supabase (running via Docker)
- UV package manager

## 🔍 Development

### Project Structure
```
src/
├── api/
│   └── routes.py              # FastAPI routes and endpoints
├── agents/                    # AI agents for analysis
├── utils/
│   ├── document_ingestion.py  # Document processing
│   ├── vector_operations.py   # Vector database operations
│   ├── memory_store.py        # Shared memory management
│   └── checkpointer.py        # State persistence
├── workflows/                 # Analysis workflows
└── config/
    └── settings.py            # Configuration management
```

### Running Tests
```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest test_rag_endpoints.py
```

## 📄 OpenAPI Schema

The complete OpenAPI 3.1.0 schema is available in multiple formats:

- **JSON**: `openapi.json`
- **YAML**: `openapi.yaml`
- **Interactive**: http://localhost:8000/docs

## 🛠️ Error Handling

The API includes comprehensive error handling:

- **400 Bad Request**: Invalid input parameters
- **404 Not Found**: Resource not found
- **413 Payload Too Large**: File exceeds size limit
- **422 Validation Error**: Request validation failed
- **500 Internal Server Error**: Server processing error

## 🔒 Security

- File uploads are validated for type and size
- Documents are stored with metadata tracking
- Client isolation through namespaces
- All uploads logged for audit trail
- Zero hallucination tolerance for financial workflows

## 📈 Performance

- Async/await support for concurrent requests
- PostgreSQL with pgvector for efficient similarity search
- Configurable chunking and embedding strategies
- Memory fallback for checkpoint resilience

---

**Version**: 1.0.0  
**Framework**: FastAPI  
**License**: Private  
**Support**: Backend API only (no web interface)