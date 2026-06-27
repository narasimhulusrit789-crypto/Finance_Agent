"""
Central configuration for the IBM watsonx.ai Finance Agent system.
All settings are loaded from environment variables (or .env file).
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class WatsonxSettings(BaseSettings):
    """IBM watsonx.ai connection settings."""

    model_config = SettingsConfigDict(env_prefix="WATSONX_", extra="ignore", env_file=".env", env_file_encoding="utf-8")

    api_key: str = Field(..., description="IBM Cloud API key")
    project_id: str = Field(..., description="watsonx.ai project ID")
    url: str = Field("https://us-south.ml.cloud.ibm.com")
    region: str = Field("us-south")

    # Granite model parameters
    model_id: str = Field("ibm/granite-4-h-small", env="GRANITE_MODEL_ID")   # IBM Granite-4.0 (production model ID)
    max_tokens: int = Field(4096, env="GRANITE_MAX_TOKENS")
    temperature: float = Field(0.7, env="GRANITE_TEMPERATURE")
    top_p: float = Field(0.95, env="GRANITE_TOP_P")


class IBMCloudSettings(BaseSettings):
    """IBM Cloud platform settings."""

    model_config = SettingsConfigDict(env_prefix="IBM_CLOUD_", extra="ignore", env_file=".env", env_file_encoding="utf-8")

    api_key: str = Field(..., description="IBM Cloud API key")
    region: str = Field("us-south")
    resource_group: str = Field("default")

    # Cloud Object Storage
    cos_api_key: str = Field("", env="IBM_COS_API_KEY")
    cos_instance_crn: str = Field("", env="IBM_COS_INSTANCE_CRN")
    cos_bucket_name: str = Field("finance-agent-files", env="IBM_COS_BUCKET_NAME")
    cos_endpoint: str = Field(
        "https://s3.us-south.cloud-object-storage.appdomain.cloud",
        env="IBM_COS_ENDPOINT",
    )


class VectorDBSettings(BaseSettings):
    """Vector store / RAG configuration."""

    model_config = SettingsConfigDict(env_prefix="VECTOR_", extra="ignore", env_file=".env", env_file_encoding="utf-8")

    db_type: Literal["chromadb", "faiss"] = Field("chromadb", env="VECTOR_DB_TYPE")
    db_path: str = Field("./data/vector_store", env="VECTOR_DB_PATH")
    collection_name: str = Field("finance_knowledge", env="VECTOR_COLLECTION_NAME")
    embedding_model: str = Field("all-MiniLM-L6-v2", env="EMBEDDING_MODEL")
    chunk_size: int = Field(512, env="CHUNK_SIZE")
    chunk_overlap: int = Field(64, env="CHUNK_OVERLAP")
    top_k_results: int = Field(5, env="TOP_K_RESULTS")


class AppSettings(BaseSettings):
    """Application-level settings."""

    model_config = SettingsConfigDict(extra="ignore", env_file=".env", env_file_encoding="utf-8")

    host: str = Field("0.0.0.0", env="APP_HOST")
    port: int = Field(8000, env="APP_PORT")
    env: Literal["development", "production", "testing"] = Field(
        "production", env="APP_ENV"
    )
    secret_key: str = Field(..., env="SECRET_KEY")
    debug: bool = Field(False, env="DEBUG")

    # Cache
    redis_url: str = Field("redis://localhost:6379", env="REDIS_URL")
    cache_ttl: int = Field(3600, env="CACHE_TTL")
    session_ttl: int = Field(86400, env="SESSION_TTL")

    # Market data API keys
    alpha_vantage_api_key: str = Field("", env="ALPHA_VANTAGE_API_KEY")
    news_api_key: str = Field("", env="NEWS_API_KEY")
    finnhub_api_key: str = Field("", env="FINNHUB_API_KEY")

    @field_validator("secret_key")
    @classmethod
    def secret_key_must_be_set(cls, v: str) -> str:
        if v in ("", "your_secret_key_here_change_in_production"):
            raise ValueError("SECRET_KEY must be set to a secure random value")
        return v


class Settings(BaseSettings):
    """Root settings — aggregates all sub-settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    watsonx: WatsonxSettings = Field(default_factory=WatsonxSettings)
    ibm_cloud: IBMCloudSettings = Field(default_factory=IBMCloudSettings)
    vector_db: VectorDBSettings = Field(default_factory=VectorDBSettings)
    app: AppSettings = Field(default_factory=AppSettings)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached application settings (loaded once at startup)."""
    return Settings()
