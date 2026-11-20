FROM python:3.12-slim

# Install uv (the package manager this project uses) via its official
# installer, which places the binary at /root/.local/bin/uv.
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* \
    && curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /app

# Copy only the dependency manifests first so Docker's layer cache skips
# `uv sync` entirely on rebuilds where only application code changed.
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-install-project

# Now copy the rest of the source and install the project itself.
COPY src/ ./src/
RUN uv sync --locked

EXPOSE 8000

# The api service uses this default command as-is; the ingest service
# overrides it (see docker-compose.yml).
CMD ["uv", "run", "uvicorn", "qna.api:app", "--host", "0.0.0.0", "--port", "8000"]
