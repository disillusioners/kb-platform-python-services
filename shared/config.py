"""Configuration for KB platform services."""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Server
    server_host: str = "0.0.0.0"
    server_port: int = 8000

    # Database
    database_url: str = "postgresql://kbuser:kbpassword@localhost:5432/kb"

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "documents"

    # OpenAI
    openai_api_key: str = ""

    # Embeddings
    embedding_model: str = "text-embedding-3-small"
    embedding_dimension: int = 1536

    # Chunking
    chunk_size: int = 512
    chunk_overlap: int = 50

    # LLM
    llm_model: str = "gpt-3.5-turbo"
    llm_temperature: float = 0.7

    # S3
    s3_bucket: str = "kb-documents"
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_region: str = "us-east-1"
    s3_endpoint_url: str | None = None

    # Temporal
    temporal_host: str = "localhost"
    temporal_port: int = 7233


@lru_cache
def get_settings() -> Settings:
    """Get cached settings."""
    return Settings()
