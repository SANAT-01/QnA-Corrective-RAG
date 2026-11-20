# QnA — Corrective RAG with LangGraph

A Document Q&A system that answers questions over your own PDFs/Markdown files, built as a learning project for [LangGraph](https://langchain-ai.github.io/langgraph/).

It implements the **Corrective RAG** pattern: retrieve documents, have the LLM grade whether they're actually relevant, and if not, rewrite the question and retry (up to a configurable number of times) before answering — rather than blindly generating an answer from whatever got retrieved.

## How it works

```
START → retrieve → grade_documents ─┬─(relevant)───────────────→ generate → END
                         ▲           └─(not relevant, retries<max)→ rewrite_query ─┘
                         └────────────────────────────────────────────┘
                         (not relevant, retries>=max) → generate (low-confidence flag) → END
```

- **retrieve** — embeds the question and pulls the top-k most similar chunks from Chroma.
- **grade_documents** — asks the LLM whether the retrieved chunks are relevant to the question.
- **rewrite_query** — if not relevant and retries remain, asks the LLM to rephrase the question for a better retrieval pass, then loops back to `retrieve`.
- **generate** — answers using the retrieved chunks, citing sources. If retries were exhausted without relevant context, the answer is prefixed with a low-confidence disclaimer.

The graph is defined in [src/qna/graph.py](src/qna/graph.py), which is also the most heavily commented file in the project if you want to learn how LangGraph's `StateGraph`, nodes, and conditional edges fit together. A diagram of the compiled graph is regenerated automatically on every run and saved to `graph.png`.

## Architecture

- **LLM & embeddings**: any OpenAI-compatible API (chat + embeddings), configured via `LLM_BASE_URL`/`LLM_API_KEY`.
- **Vector store**: a standalone [Chroma](https://www.trychroma.com/) server (not embedded), reached over HTTP.
- **Ingestion** ([src/qna/ingest.py](src/qna/ingest.py)): incremental and idempotent. A local manifest (`manifest.json`) tracks each file's content hash and the chunk IDs it produced, so re-running ingestion only re-embeds files that are new or changed, and cleans up chunks for files that were deleted. Safe to interrupt and re-run at any point.
- **API** ([src/qna/api.py](src/qna/api.py)): a FastAPI service exposing `POST /ask` and `GET /health`.
- **CLI** ([src/qna/main.py](src/qna/main.py)): `uv run python -m qna.main "question"` for quick one-off questions.

Three containers in Docker Compose: `chroma` (vector store), `api` (FastAPI service), `ingest` (one-shot ingestion job) — see [docker-compose.yml](docker-compose.yml).

## Setup

1. Copy the env template and fill in your LLM credentials:
   ```bash
   cp .env.example .env
   ```
2. Drop your PDF/Markdown files into `docs/`.

### Run with Docker Compose (recommended)

```bash
# Start Chroma and the API
docker compose up --build -d chroma api

# Ingest your documents (re-run this any time docs/ changes)
docker compose run --rm ingest

# Ask a question
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the OSI model?"}'
```

- Health check: `curl http://localhost:8000/health`
- Interactive API docs: http://localhost:8000/docs
- Stop everything: `docker compose down` (add `-v` to also wipe the Chroma/manifest volumes)

> If the Chroma volume ever gets reset independently of the ingestion manifest (e.g. after `docker compose down -v` on just one service, or restoring from a partial backup), the manifest will think files are already ingested and skip them, leaving the collection empty. Run `docker compose run --rm ingest uv run python -m qna.ingest --rebuild` to force a full re-embed from scratch.

### Run locally without Docker

Chroma always runs as a standalone server, so start one first:

```bash
docker run -p 8001:8000 -v chroma_data:/chroma/chroma chromadb/chroma
```

Then, from the project root:

```bash
uv run python -m qna.ingest              # ingest docs/
uv run python -m qna.main "your question"  # CLI
uv run uvicorn qna.api:app --reload        # API on localhost:8000
```

## Configuration

All settings are centralized in [src/qna/config.py](src/qna/config.py) and read from `.env`. See [.env.example](.env.example) for the full list, including `CHUNK_SIZE`, `CHUNK_OVERLAP`, `MAX_RETRIES`, `TOP_K`, and `LOG_LEVEL`.

## Project status

This project is built incrementally, version by version — see `main` for the current state. Implemented so far:

- Core self-correcting retrieve/grade/rewrite/generate loop
- Incremental, idempotent ingestion with crash recovery
- FastAPI service layer with structured logging, centralized config, and input validation
- Docker Compose setup with a standalone Chroma server

Planned next: resilience (retry/backoff on LLM calls, embedding throttling) and observability (richer trace reasoning), followed by tests, auth, async conversion, streaming, and CI.
