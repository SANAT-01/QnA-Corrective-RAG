"""Factory functions for the chat model and embeddings client.

Both point at the same self-hosted, OpenAI-compatible API (chat completions
and /v1/embeddings), configured via environment variables loaded from .env.
"""

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()


def get_chat_model() -> ChatOpenAI:
    return ChatOpenAI(
        base_url=os.environ["LLM_BASE_URL"],
        api_key=os.environ["LLM_API_KEY"],
        model=os.environ.get("LLM_MODEL", "auto"),
        temperature=0,
    )


def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        base_url=os.environ["LLM_BASE_URL"],
        api_key=os.environ["LLM_API_KEY"],
        model=os.environ.get("EMBEDDING_MODEL", "auto"),
        # Many self-hosted OpenAI-compatible APIs only accept raw text input,
        # not the tiktoken-encoded integer arrays langchain sends by default.
        check_embedding_ctx_length=False,
    )
