"""Factory functions for the chat model, embeddings client, and vector store.

Chat/embeddings point at the self-hosted, OpenAI-compatible API (chat
completions and /v1/embeddings). The vector store talks to a standalone
Chroma server over HTTP. All configured via qna.config.settings.
"""

import chromadb
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from qna.config import settings


def get_chat_model() -> ChatOpenAI:
    return ChatOpenAI(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        temperature=0,
    )


def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=settings.embedding_model,
        # Many self-hosted OpenAI-compatible APIs only accept raw text input,
        # not the tiktoken-encoded integer arrays langchain sends by default.
        check_embedding_ctx_length=False,
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
