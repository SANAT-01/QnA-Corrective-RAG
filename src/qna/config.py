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

    # --- Vector store ---
    # v0.4+: always a standalone Chroma server reached over HTTP (embedded
    # file-mode from v0.1-v0.3 is gone). For non-Docker use, run one
    # yourself first, e.g. `docker run -p 8001:8000 chromadb/chroma`.
    # docker-compose.yml overrides chroma_host to "chroma" (the service
    # name on the compose network) and chroma_port to 8000 (its internal
    # port - not the host-published port, since chroma isn't published).
    chroma_host: str = "localhost"
    chroma_port: int = 8001
    collection_name: str = "qna_docs"

    # --- Ingestion ---
    docs_dir: str = "docs"
    # Where ingest.py's manifest.json lives. Local default; overridden to
    # /data by docker-compose.yml, pointing at a dedicated volume shared
    # between the api and ingest containers.
    manifest_dir: str = "chroma_db"
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
