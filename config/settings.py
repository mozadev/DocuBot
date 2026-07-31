"""
Application configuration.

Everything tunable lives here and comes from the environment, so the same image
runs in every environment with no code change. Pydantic validates at import
time, which means a missing API key fails at startup with a clear message rather
than on the first user request.
"""

from __future__ import annotations

import os

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        case_sensitive=False,
    )

    # --- OpenAI ---
    openai_api_key: str = Field(..., alias="OPENAI_API_KEY")
    openai_model: str = Field("gpt-4o-mini", alias="OPENAI_MODEL")
    # Low but not zero. 0.0 makes the model repeat awkward phrasing verbatim from
    # the source text; 0.2 keeps answers grounded while letting it write a sentence.
    openai_temperature: float = Field(0.2, alias="OPENAI_TEMPERATURE")

    # --- Embeddings and chunking ---
    embedding_model: str = Field("text-embedding-3-small", alias="EMBEDDING_MODEL")
    chunk_size: int = Field(1000, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(200, alias="CHUNK_OVERLAP")

    # --- Vector store ---
    lancedb_path: str = Field("./data/vector_db", alias="LANCE_DB_PATH")
    lancedb_table: str = Field("documents", alias="LANCE_DB_TABLE")

    # --- Multimodal ingest ---
    enable_multimodal: bool = Field(True, alias="ENABLE_MULTIMODAL")
    vision_model: str = Field("gpt-4o", alias="VISION_MODEL")
    images_path: str = Field("./data/images", alias="IMAGES_PATH")
    min_image_size: int = Field(100, alias="MIN_IMAGE_SIZE")

    # --- Quality and guardrails ---
    min_grounding_score: float = Field(0.25, alias="MIN_GROUNDING_SCORE")
    max_upload_mb: int = Field(25, alias="MAX_UPLOAD_MB")

    # --- Cache and observability ---
    cache_max_entries: int = Field(1000, alias="CACHE_MAX_ENTRIES")
    cache_ttl_seconds: int = Field(3600, alias="CACHE_TTL_SECONDS")
    max_traces: int = Field(500, alias="MAX_TRACES")

    # --- API ---
    # Comma-separated origins. Defaults to the local Streamlit UI; set explicitly
    # in production rather than widening to "*".
    cors_origins: str = Field("http://localhost:8501", alias="CORS_ORIGINS")

    # --- App ---
    app_name: str = Field("DocuBot AI", alias="APP_NAME")
    app_version: str = Field("2.0.0", alias="APP_VERSION")
    debug: bool = Field(False, alias="DEBUG")

    # --- Logging ---
    log_level: str = Field("INFO", alias="LOG_LEVEL")
    log_file: str = Field("./logs/app.log", alias="LOG_FILE")

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


settings = Settings()

os.makedirs(settings.lancedb_path, exist_ok=True)
os.makedirs(settings.images_path, exist_ok=True)
os.makedirs(os.path.dirname(settings.log_file) or ".", exist_ok=True)
