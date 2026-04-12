"""All settings classes. Pydantic-settings v2 — Field(env=...) is NOT valid here."""
from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class KafkaSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="KAFKA_", env_file=".env", extra="ignore")

    BOOTSTRAP_SERVERS: str = "localhost:9092"
    RAW_TICKS_TOPIC: str = "raw.ticks"
    FEATURES_TOPIC: str = "features"
    DLQ_TOPIC: str = "dlq"
    CONSUMER_GROUP: str = "stock-platform"


class RedisSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="REDIS_", env_file=".env", extra="ignore")

    URL: str = "redis://localhost:6379/0"
    FEATURE_TTL_SECONDS: int = 3600


class IngestionSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="INGESTION_", env_file=".env", extra="ignore")

    URL: str = "http://localhost:8001"
    POLL_INTERVAL_SECONDS: int = 60


class FeatureSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FEATURE_", env_file=".env", extra="ignore")

    S3_BUCKET: str = "stock-features"
    S3_PREFIX: str = "features"


class InferenceSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="INFERENCE_", env_file=".env", extra="ignore")

    URL: str = "http://localhost:8002"
    MODEL_RELOAD_INTERVAL: int = 300


class LLMSettings(BaseSettings):
    """
    Pydantic-settings v2: secrets come from the real environment.
    Use validation_alias so the field reads the plain env-var name
    (without the LLM_ prefix that model_config would otherwise prepend).
    """
    model_config = SettingsConfigDict(env_prefix="LLM_", env_file=".env", extra="ignore")

    LOG_LEVEL: str = "INFO"
    LLM_MODEL: str = "claude-sonnet-4-6"
    LLM_MAX_TOKENS: int = 1024
    PINECONE_INDEX_NAME: str = "stock-news-embeddings"
    RAG_TOP_K: int = 6
    RAG_MIN_RELEVANCE: float = 0.72
    EMBED_MODEL: str = "text-embedding-3-small"

    # Secrets — read directly from env (no prefix) via validation_alias
    ANTHROPIC_API_KEY: str = Field(default="", validation_alias="ANTHROPIC_API_KEY")
    PINECONE_API_KEY: str = Field(default="", validation_alias="PINECONE_API_KEY")
    NEWS_API_KEY: str = Field(default="", validation_alias="NEWS_API_KEY")


@lru_cache
def get_kafka_settings() -> KafkaSettings:
    return KafkaSettings()


@lru_cache
def get_redis_settings() -> RedisSettings:
    return RedisSettings()


@lru_cache
def get_ingestion_settings() -> IngestionSettings:
    return IngestionSettings()


@lru_cache
def get_feature_settings() -> FeatureSettings:
    return FeatureSettings()


@lru_cache
def get_inference_settings() -> InferenceSettings:
    return InferenceSettings()


@lru_cache
def get_llm_settings() -> LLMSettings:
    return LLMSettings()
