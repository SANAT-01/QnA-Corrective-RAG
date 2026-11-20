"""Centralized configuration.

Every setting that used to be a scattered module-level constant or a raw
os.environ[...] lookup (in llm.py, graph.py, ingest.py) now lives here as
one pydantic-settings Settings instance, loaded once from .env.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- LLM / embeddings (self-hosted OpenAI-compatible API) ---
    llm_base_url: str
    llm_api_key: str
    llm_model: str = "auto"
    embedding_model: str = "auto"

    # --- Vector store (embedded Chroma through v0.3; replaced by a
    # standalone server reached over HTTP starting v0.4) ---
    chroma_dir: str = "chroma_db"
    collection_name: str = "qna_docs"

    # --- Ingestion ---
    docs_dir: str = "docs"
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # --- Graph behavior ---
    max_retries: int = 2
    top_k: int = 4

    # --- Service ---
    log_level: str = "INFO"
    api_host: str = "0.0.0.0"
    api_port: int = 8000


# Built once at import time; every module imports this same instance
# instead of reading the environment or defining its own constants.
settings = Settings()
