"""FastAPI service exposing the Corrective RAG graph over HTTP.

Run with: uv run uvicorn qna.api:app --reload
"""

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator

from qna import logging_config  # noqa: F401 - configures logging on import
from qna.graph import app as graph_app
from qna.graph import source_label

logger = logging.getLogger(__name__)

app = FastAPI(title="QnA - Corrective RAG API")

MAX_QUESTION_LENGTH = 2000


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    # Never leak internal exception text/stack traces to API clients - the
    # real detail goes to the server logs only (see ask() below).
    return JSONResponse(
        status_code=500,
        content={"detail": "Something went wrong processing your question"},
    )


class AskRequest(BaseModel):
    question: str

    @field_validator("question")
    @classmethod
    def question_must_be_meaningful(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("question must not be empty")
        if len(stripped) > MAX_QUESTION_LENGTH:
            raise ValueError(f"question must be at most {MAX_QUESTION_LENGTH} characters")
        return stripped


class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    confident: bool
    retries: int
    trace: list[str]


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    initial_state = {
        "question": request.question,
        "documents": [],
        "retry_count": 0,
        "relevant": False,
        "confident": False,
        "generation": "",
        "trace": [],
    }

    # Any exception here (a GenerationError from the chat model, a Chroma
    # failure, anything else) is deliberately not surfaced to the client -
    # the server log gets the full detail, the API response stays generic
    # so we never leak internal exception text/stack traces to callers.
    try:
        result = graph_app.invoke(initial_state)
    except Exception:
        logger.exception("ask: graph invocation failed")
        raise

    sources = []
    for doc in result["documents"]:
        label = source_label(doc)
        if label not in sources:
            sources.append(label)

    return AskResponse(
        answer=result["generation"],
        sources=sources,
        confident=result["confident"],
        retries=result["retry_count"],
        trace=result["trace"],
    )
