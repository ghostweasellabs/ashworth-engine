from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # LLM Configuration
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    llm_provider: str = "openai"
    embedding_model: str = "text-embedding-ada-002"
    
    # GPT Model Configuration
    primary_model: str = "gpt-4.1"  # For complex analysis and report generation
    secondary_model: str = "gpt-4.1-mini"  # For simple tasks and validation
    fallback_model: str = "gpt-4o"  # Emergency fallback
    
    # Model routing configuration
    use_mini_for_classification: bool = True
    use_primary_for_reports: bool = True
    max_tokens_primary: int = 32768
    max_tokens_secondary: int = 16384
    temperature: float = 0.1
    
    # Supabase Configuration
    supabase_url: str = "http://127.0.0.1:54321"
    supabase_anon_key: Optional[str] = None
    supabase_service_key: Optional[str] = None
    supabase_jwt_secret: Optional[str] = None
    
    # Database Configuration (PostgreSQL with pgvector)
    database_url: str = "postgresql://postgres:postgres@127.0.0.1:54322/postgres"
    vecs_connection_string: str = "postgresql://postgres:postgres@127.0.0.1:54322/postgres"
    
    # Storage Configuration
    storage_provider: str = "supabase"
    storage_bucket: str = "reports"
    charts_bucket: str = "charts"
    supabase_storage_url: str = "http://127.0.0.1:54321/storage/v1"
    
    # Vector Database Configuration
    vector_collection_name: str = "documents"
    vector_dimension: int = 1536
    vector_similarity_threshold: float = 0.8
    
    # RAG Configuration
    rag_enabled: bool = True
    rag_chunk_size: int = 1000
    rag_chunk_overlap: int = 200
    rag_top_k: int = 5
    rag_score_threshold: float = 0.7
    rag_collection_names: dict = {
        "irs_documents": "irs_documents",
        "financial_regulations": "financial_regulations", 
        "tax_guidance": "tax_guidance",
        "user_documents": "user_documents"
    }
    
    # Memory and Persistence Configuration  
    enable_checkpointing: bool = True
    enable_shared_memory: bool = True
    memory_ttl_hours: int = 168  # 7 days
    checkpoint_retention_days: int = 30
    
    # Store Configuration for Shared Memory
    store_index_config: dict = {
        "dims": 1536,
        "fields": ["$"]  # Embed entire document by default
    }
    
    # API Configuration
    api_auth_key: Optional[str] = None
    ae_env: str = "development"
    
    # Processing Configuration
    max_upload_size: int = 52428800  # 50MB
    report_retention_days: int = 90
    ocr_language: str = "eng"
    
    # Document Ingestion Configuration
    supported_doc_types: list = ["pdf", "txt", "md", "docx", "csv", "xlsx"]
    max_doc_size_mb: int = 50
    ingestion_batch_size: int = 100
    
    # Performance
    max_concurrent_requests: int = 5
    llm_timeout_seconds: int = 300
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "json"
    
    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()