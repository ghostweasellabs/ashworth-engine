from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # LLM Configuration
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    llm_provider: str = "openai"
    embedding_model: str = "text-embedding-ada-002"
    
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
    
    # API Configuration
    api_auth_key: Optional[str] = None
    ae_env: str = "development"
    
    # Processing Configuration
    max_upload_size: int = 52428800  # 50MB
    report_retention_days: int = 90
    ocr_language: str = "eng"
    
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