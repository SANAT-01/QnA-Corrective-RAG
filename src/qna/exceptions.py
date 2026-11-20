"""Custom exceptions, raised at real failure boundaries instead of letting
raw exceptions (requests/openai/chromadb internals) leak to callers.
"""


class IngestionError(Exception):
    """Raised when docs/ can't be read, or a file fails to load/split."""


class GenerationError(Exception):
    """Raised when a call to the chat model fails (grading, rewriting, or
    answering)."""


class LLMError(Exception):
    """Raised when a chat completion call fails after retries are
    exhausted, or fails immediately on a non-transient (4xx) error."""


class EmbeddingError(Exception):
    """Raised when an embeddings call fails after retries are exhausted,
    or fails immediately on a non-transient (4xx) error."""
