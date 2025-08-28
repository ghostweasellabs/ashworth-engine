"""Configuration management using Pydantic settings."""

import os
from typing import Optional
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
        default="",
        env="SUPABASE_ANON_KEY"
    )
    supabase_service_key: str = Field(
        default="",
        env="SUPABASE_SERVICE_KEY"
    )
    
    # LLM Configuration
    llm_provider: str = Field(
        default="ollama",
        env="LLM_PROVIDER"
    )
    ollama_host: str = Field(
        default="http://192.168.7.43:11434",
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
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global settings instance
settings = Settings()