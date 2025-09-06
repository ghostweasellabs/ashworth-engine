"""Configuration management using Pydantic settings."""

import os
from typing import Optional, List
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Database Configuration
    database_url: str = Field(
        default="postgresql://postgres:postgres@localhost:5432/ashworth",
        env="DATABASE_URL"
    )
    supabase_url: str = Field(
        default="http://localhost:54321",
        env="SUPABASE_URL"
    )
    supabase_anon_key: str = Field(
        default="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZS1kZW1vIiwicm9sZSI6ImFub24iLCJleHAiOjE5ODM4MTI5OTZ9.CRXP1A7WOeoJeXxjNni43kdQwgnWNReilDMblYTn_I0",
        env="SUPABASE_ANON_KEY"
    )
    supabase_service_key: str = Field(
        default="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZS1kZW1vIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImV4cCI6MTk4MzgxMjk5Nn0.EGIM96RAZx35lJzdJsyH-qQwv8Hdp7fsn3W0YpN81IU",
        env="SUPABASE_SERVICE_KEY"
    )
    
    # LLM Configuration
    llm_provider: str = Field(
        default="ollama",
        env="LLM_PROVIDER"
    )
    ollama_host: str = Field(
        default="http://192.168.1.220:11434",
        env="OLLAMA_HOST"
    )
    openai_api_key: Optional[str] = Field(
        default=None,
        env="OPENAI_API_KEY"
    )
    google_api_key: Optional[str] = Field(
        default=None,
        env="GOOGLE_API_KEY"
    )
    
    # LLM Router Configuration
    llm_fallback_order: List[str] = Field(
        default=["ollama", "openai", "google"],
        env="LLM_FALLBACK_ORDER"
    )
    
    # Model Selection Configuration
    heavy_model_preference: str = Field(
        default="auto",  # auto, gpt-5, gpt-4.1, gemini-2.5-pro, gpt-oss:20b
        env="HEAVY_MODEL_PREFERENCE"
    )
    light_model_preference: str = Field(
        default="auto",  # auto, gpt-4.1-mini, gemini-2.5-flash, gemma3n:e4b
        env="LIGHT_MODEL_PREFERENCE"
    )
    
    # GPT-5 Specific Configuration
    gpt5_reasoning_effort: str = Field(
        default="medium",
        env="GPT5_REASONING_EFFORT"
    )
    gpt5_include_reasoning: bool = Field(
        default=False,
        env="GPT5_INCLUDE_REASONING"
    )
    gpt5_reasoning_depth: str = Field(
        default="standard",
        env="GPT5_REASONING_DEPTH"
    )
    
    # Performance Monitoring Configuration
    enable_metrics_logging: bool = Field(
        default=True,
        env="ENABLE_METRICS_LOGGING"
    )
    metrics_log_file: str = Field(
        default="logs/llm_metrics.jsonl",
        env="METRICS_LOG_FILE"
    )
    performance_threshold_seconds: float = Field(
        default=30.0,
        env="PERFORMANCE_THRESHOLD_SECONDS"
    )
    cost_threshold_dollars: float = Field(
        default=0.10,
        env="COST_THRESHOLD_DOLLARS"
    )
    
    # Provider Health Check Configuration
    health_check_interval_seconds: int = Field(
        default=300,  # 5 minutes
        env="HEALTH_CHECK_INTERVAL_SECONDS"
    )
    provider_timeout_seconds: int = Field(
        default=60,
        env="PROVIDER_TIMEOUT_SECONDS"
    )
    max_retries: int = Field(
        default=3,
        env="MAX_RETRIES"
    )
    retry_delay_seconds: float = Field(
        default=1.0,
        env="RETRY_DELAY_SECONDS"
    )
    
    # Application Configuration
    debug: bool = Field(
        default=True,
        env="DEBUG"
    )
    log_level: str = Field(
        default="INFO",
        env="LOG_LEVEL"
    )
    environment: str = Field(
        default="development",
        env="ENVIRONMENT"
    )
    
    # Security
    secret_key: str = Field(
        default="dev-secret-key-change-in-production",
        env="SECRET_KEY"
    )
    
    # File Upload Configuration
    max_file_size: int = Field(
        default=52428800,  # 50MB
        env="MAX_FILE_SIZE"
    )
    upload_dir: str = Field(
        default="uploads",
        env="UPLOAD_DIR"
    )

    # Embedding Model Configuration
    embedding_model: str = Field(
        default="embeddinggemma",
        env="EMBEDDING_MODEL"
    )
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get the global settings instance."""
    return settings