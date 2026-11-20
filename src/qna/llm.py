"""Factory functions for the chat model, embeddings client, and vector store.

Chat/embeddings point at the self-hosted, OpenAI-compatible API (chat
completions and /v1/embeddings). The vector store talks to a standalone
Chroma server over HTTP. All configured via qna.config.settings.
"""

import chromadb
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from qna.config import settings
from qna.exceptions import EmbeddingError, LLMError
from qna.resilience import with_llm_retry

# Request timeout per attempt. Paired with with_llm_retry's 3-attempt
# backoff: a single hung request can't block forever, and still gets a
# couple of chances to succeed before we give up.
_REQUEST_TIMEOUT_SECONDS = 30


class ResilientChatOpenAI(ChatOpenAI):
    """ChatOpenAI whose .invoke() is wrapped in the shared retry policy and
    translates any failure (after retries are exhausted, or immediately for
    a non-transient error) into a clear LLMError."""

    def invoke(self, *args, **kwargs):
        try:
            return with_llm_retry(super().invoke)(*args, **kwargs)
        except Exception as e:
            raise LLMError(f"Chat completion failed: {e}") from e


class ResilientEmbeddings(OpenAIEmbeddings):
    """OpenAIEmbeddings whose embed_documents/embed_query are wrapped in the
    shared retry policy and translate failures into a clear EmbeddingError.

    Both ingestion (via Chroma's add_documents, which calls embed_documents
    in batches) and retrieval (via similarity_search, which calls
    embed_query) go through this same resilient path.
    """

    def embed_documents(self, texts: list[str], chunk_size: int | None = None) -> list[list[float]]:
        try:
            return with_llm_retry(super().embed_documents)(texts, chunk_size=chunk_size)
        except Exception as e:
            raise EmbeddingError(f"Embedding call failed: {e}") from e

    def embed_query(self, text: str) -> list[float]:
        try:
            return with_llm_retry(super().embed_query)(text)
        except Exception as e:
            raise EmbeddingError(f"Embedding call failed: {e}") from e


def get_chat_model() -> ChatOpenAI:
    return ResilientChatOpenAI(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        temperature=0,
        timeout=_REQUEST_TIMEOUT_SECONDS,
    )


def get_embeddings() -> OpenAIEmbeddings:
    return ResilientEmbeddings(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=settings.embedding_model,
        # Many self-hosted OpenAI-compatible APIs only accept raw text input,
        # not the tiktoken-encoded integer arrays langchain sends by default.
        check_embedding_ctx_length=False,
        timeout=_REQUEST_TIMEOUT_SECONDS,
    )


def get_vector_store() -> Chroma:
    """Open a handle to the standalone Chroma server (v0.4+).

    graph.py uses this for retrieval, ingest.py uses it for writing. Both
    get a Chroma wrapper around the same HttpClient connection, so they're
    always talking to the same remote collection - no local files involved
    on this side.
    """
    client = chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port)
    return Chroma(
        client=client,
        collection_name=settings.collection_name,
        embedding_function=get_embeddings(),
    )
