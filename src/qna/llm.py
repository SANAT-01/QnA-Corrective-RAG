"""Factory functions for the chat model and embeddings client.

Both point at the same self-hosted, OpenAI-compatible API (chat completions
and /v1/embeddings), configured via qna.config.settings.
"""

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
