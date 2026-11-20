"""Custom exceptions, raised at real failure boundaries instead of letting
raw exceptions (requests/openai/chromadb internals) leak to callers.
"""


class IngestionError(Exception):
    """Raised when docs/ can't be read, or a file fails to load/split."""


class GenerationError(Exception):
    """Raised when a call to the chat model fails (grading, rewriting, or
    answering)."""
