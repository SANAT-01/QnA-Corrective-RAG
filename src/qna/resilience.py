"""Retry policy for calls to the external LLM API (chat + embeddings).

langchain-openai's ChatOpenAI/OpenAIEmbeddings raise exceptions from the
underlying `openai` Python client. Those exceptions already distinguish
transient failures (connection drop, timeout, 5xx) from permanent ones
(401 bad key, 400 bad request, etc.) - we just need to tell tenacity which
of those is worth retrying.

Retrying a permanent error wastes time: a bad API key will fail the same
way on every attempt, so only transient failures get retried here.
"""

import logging

from openai import APIConnectionError, APITimeoutError, InternalServerError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

# Connection drops and timeouts are network-level; InternalServerError covers
# HTTP 5xx responses from the API itself. None of these mean the request
# itself was wrong - retrying has a real chance of succeeding.
_TRANSIENT_ERRORS = (APIConnectionError, APITimeoutError, InternalServerError)

# 3 attempts total, exponential backoff starting at 1s (1s, 2s, 4s between
# attempts). Logs a warning before each retry so retries are visible in
# server logs, not silent.
with_llm_retry = retry(
    retry=retry_if_exception_type(_TRANSIENT_ERRORS),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=4),
    reraise=True,
    before_sleep=lambda retry_state: logger.warning(
        "Retrying after transient error (attempt %d): %s",
        retry_state.attempt_number,
        retry_state.outcome.exception(),
    ),
)
