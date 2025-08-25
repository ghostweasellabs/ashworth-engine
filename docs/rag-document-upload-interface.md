# RAG Document Upload Interface

This document describes the new RAG (Retrieval-Augmented Generation) document upload interface that has been added to the Ashworth Engine v2 API.

## Overview

The RAG document upload interface allows you to upload and ingest documents into the system's knowledge base, which can then be used by the AI agents for enhanced analysis and recommendations.

## Available Endpoints

### 1. Upload Document File
**POST** `/rag/documents/upload`

Upload a file to be ingested into the RAG system.

**Parameters:**
- `file` (file): The document file to upload
- `collection_name` (string, optional): Target collection (default: "user_documents")
- `namespace` (string, optional): Namespace for organization (default: "default")
- `client_id` (string, optional): Client identifier
- `document_type` (string, optional): Type of document
- `description` (string, optional): Document description

**Supported File Types:**
- PDF (`.pdf`)
- Text files (`.txt`)
- Markdown (`.md`)
- Word documents (`.docx`)
- CSV files (`.csv`)
- Excel files (`.xlsx`, `.xls`)

**Response:**
```json
{
  "document_id": "uuid",
  "status": "success",
  "filename": "document.pdf",
  "collection_name": "user_documents",
  "namespace": "default",
  "chunks_created": 5,
  "ingestion_timestamp": "2024-01-01T12:00:00Z",
  "file_size_bytes": 1024000,
  "metadata": {
    "chunk_ids": ["id1", "id2", "..."],
    "storage_path": "rag_documents/..."
  }
}
```

### 2. Upload Text Content
**POST** `/rag/documents/upload-text`

Upload raw text content directly to the RAG system.

**Parameters:**
- `text` (string): The text content to upload
- `title` (string): Title for the document
- `collection_name` (string, optional): Target collection (default: "user_documents")
- `namespace` (string, optional): Namespace for organization (default: "default")
- `client_id` (string, optional): Client identifier
- `document_type` (string, optional): Type of document
- `description` (string, optional): Document description

### 3. Search Documents
**POST** `/rag/search`

Search for documents in the RAG system using semantic similarity.

**Parameters:**
- `query` (string): Search query
- `collection_name` (string, optional): Collection to search (default: "user_documents")
- `namespace` (string, optional): Namespace to search (default: "default")
- `top_k` (integer, optional): Number of results to return (default: 5)
- `score_threshold` (float, optional): Minimum similarity score (default: 0.7)

**Response:**
```json
{
  "query": "business expense deductions",
  "results": [
    {
      "content": "Business expenses are generally deductible...",
      "metadata": {
        "source": "tax_guide.pdf",
        "page": 1,
        "chunk_id": 0
      },
      "similarity_score": 0.95
    }
  ],
  "collection_name": "user_documents",
  "namespace": "default",
  "search_timestamp": "2024-01-01T12:00:00Z"
}
```

### 4. Setup IRS Knowledge Base
**POST** `/rag/setup-irs-knowledge`

Initialize the default IRS knowledge base with essential tax guidance.

**Response:**
```json
{
  "status": "success",
  "message": "IRS knowledge base setup completed",
  "documents_ingested": 25,
  "setup_timestamp": "2024-01-01T12:00:00Z"
}
```

### 5. Get Collections Info
**GET** `/rag/collections`

Get information about available RAG collections and configuration.

**Response:**
```json
{
  "collections": {
    "irs_documents": "irs_documents",
    "financial_regulations": "financial_regulations",
    "tax_guidance": "tax_guidance",
    "user_documents": "user_documents"
  },
  "configuration": {
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "top_k_default": 5,
    "score_threshold_default": 0.7,
    "max_file_size_mb": 50.0,
    "supported_file_types": ["pdf", "txt", "md", "docx", "csv", "xlsx"]
  }
}
```

## Collections

The system organizes documents into different collections:

- **`irs_documents`**: Official IRS publications and forms
- **`financial_regulations`**: Financial regulatory documents
- **`tax_guidance`**: Tax optimization strategies and guidance
- **`user_documents`**: User-uploaded documents (default)

## Namespaces

Namespaces provide additional organization within collections:
- **`default`**: Default namespace for general documents
- **`essential_knowledge`**: Core IRS knowledge base
- **`client_specific`**: Client-specific documents
- **Custom namespaces**: You can specify any namespace string

## Usage Examples

### Upload a PDF Document
```bash
curl -X POST "http://localhost:8000/rag/documents/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@tax_guide.pdf" \
  -F "collection_name=user_documents" \
  -F "namespace=tax_guides" \
  -F "client_id=client_123" \
  -F "document_type=tax_guidance" \
  -F "description=Comprehensive tax deduction guide"
```

### Upload Text Content
```bash
curl -X POST "http://localhost:8000/rag/documents/upload-text" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "text=Business meals are generally 50% deductible according to IRS rules." \
  -d "title=Business Meal Deduction Rule" \
  -d "collection_name=tax_guidance" \
  -d "document_type=tax_rule"
```

### Search Documents
```bash
curl -X POST "http://localhost:8000/rag/search" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "query=business meal deduction rules" \
  -d "collection_name=tax_guidance" \
  -d "top_k=3" \
  -d "score_threshold=0.8"
```

## Integration with Analysis Workflow

Documents uploaded through this interface are automatically available to the RAG-enhanced agents:

1. **Tax Categorizer**: Uses IRS guidance for complex transaction categorization
2. **Report Generator**: Incorporates uploaded documents for context and recommendations

## Error Handling

The API includes comprehensive error handling:

- **413 Payload Too Large**: File exceeds size limit (50MB default)
- **400 Bad Request**: Invalid file type or collection name
- **500 Internal Server Error**: Processing or database errors

## Configuration

Key configuration settings in `src/config/settings.py`:

```python
# RAG Configuration
rag_enabled: bool = True
rag_chunk_size: int = 1000
rag_chunk_overlap: int = 200
rag_top_k: int = 5
rag_score_threshold: float = 0.7

# Document limits
max_upload_size: int = 52428800  # 50MB
supported_doc_types: list = ["pdf", "txt", "md", "docx", "csv", "xlsx"]
```

## Testing

To test the endpoints:

1. Start the API server:
   ```bash
   uvicorn src.api.routes:app --reload
   ```

2. Access the interactive documentation at: http://localhost:8000/docs

3. Run the test script:
   ```bash
   python test_rag_endpoints.py
   ```

## Security Considerations

- File uploads are validated for type and size
- Documents are stored with metadata tracking
- Client isolation through namespaces
- All uploads logged for audit trail

## Future Enhancements

Planned improvements include:
- Web-based upload interface
- Batch document upload
- Document versioning
- Advanced search filters
- Document management dashboard