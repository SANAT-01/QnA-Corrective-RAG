"""Load documents from docs/, split, embed, and write them to a standalone
Chroma server (reached over HTTP via qna.llm.get_vector_store()).

v0.2: incremental, idempotent ingestion. A local manifest file
(settings.manifest_dir/manifest.json) tracks, per file (keyed by its path
relative to docs/), the content hash and the Chroma chunk ids it produced.
Each run:

  - unchanged files (hash still matches the manifest) are skipped entirely
  - new files are split, embedded, and added
  - changed files (hash mismatch) have their old chunks deleted first, then
    are re-embedded as if new
  - files removed from docs/ have their chunks deleted and their manifest
    entry dropped

Every file is processed as delete-then-insert, with its manifest entry
written only *after* its chunks are confirmed in Chroma. That makes a run
that's interrupted partway through safe to just re-run: a file that never
got its manifest entry updated will simply be treated as "changed" again
next time (delete-then-insert is a no-op on the delete side if nothing
was actually written last time, and otherwise cleans up any partial chunks).
"""

import argparse
import hashlib
import json
import logging
import uuid
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from qna import logging_config  # noqa: F401 - configures logging on import
from qna.config import settings
from qna.exceptions import IngestionError
from qna.llm import get_vector_store

logger = logging.getLogger(__name__)

MARKDOWN_HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3")]


def _docs_dir() -> Path:
    return Path(settings.docs_dir)


def _manifest_dir() -> Path:
    return Path(settings.manifest_dir)


def _manifest_path() -> Path:
    return _manifest_dir() / "manifest.json"


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_manifest() -> dict:
    manifest_path = _manifest_path()
    if manifest_path.exists():
        return json.loads(manifest_path.read_text())
    return {}


def _save_manifest(manifest: dict) -> None:
    _manifest_dir().mkdir(parents=True, exist_ok=True)
    _manifest_path().write_text(json.dumps(manifest, indent=2))


def _load_and_split_file(path: Path, relative_path: str) -> list[Document]:
    """Load a single file and split it into chunks, tagging each chunk's
    metadata with its source path (relative to docs/) plus, where
    available, page number (PDFs) or nearest heading (markdown) - used for
    citations starting in v0.5.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
    )

    try:
        if path.suffix.lower() == ".pdf":
            pages = PyPDFLoader(str(path)).load()
            chunks = splitter.split_documents(pages)
            for chunk in chunks:
                # PyPDFLoader's own metadata already has a 0-indexed "page";
                # surface a human-friendly 1-indexed version too.
                chunk.metadata["source"] = relative_path
                chunk.metadata["page"] = chunk.metadata.get("page", 0) + 1
            return chunks

        if path.suffix.lower() in (".md", ".markdown"):
            text = path.read_text()
            # First split on markdown headings, so each piece carries its
            # nearest h1/h2/h3 in metadata; then re-split further by size,
            # since a section between headings can still be too long.
            header_splitter = MarkdownHeaderTextSplitter(
                headers_to_split_on=MARKDOWN_HEADERS, strip_headers=False
            )
            sections = header_splitter.split_text(text)
            chunks = splitter.split_documents(sections)
            for chunk in chunks:
                chunk.metadata["source"] = relative_path
                heading = (
                    chunk.metadata.get("h3")
                    or chunk.metadata.get("h2")
                    or chunk.metadata.get("h1")
                )
                if heading:
                    chunk.metadata["heading"] = heading
            return chunks

        # Plain text fallback.
        docs = TextLoader(str(path)).load()
        chunks = splitter.split_documents(docs)
        for chunk in chunks:
            chunk.metadata["source"] = relative_path
        return chunks
    except Exception as e:
        raise IngestionError(f"Failed to load/split {relative_path}: {e}") from e


def ingest(rebuild: bool = False) -> None:
    docs_dir = _docs_dir()

    if not docs_dir.exists():
        raise IngestionError(f"{docs_dir}/ does not exist.")
    if not any(docs_dir.iterdir()):
        logger.info("No documents found in %s/. Add some PDFs or markdown files first.", docs_dir)
        return

    vector_store = get_vector_store()

    if rebuild:
        # Wipe the remote collection and the local manifest, then fall
        # through to the normal ingest loop below, which will treat every
        # file in docs/ as new.
        vector_store.delete_collection()
        vector_store = get_vector_store()
        if _manifest_path().exists():
            _manifest_path().unlink()

    manifest = _load_manifest()

    current_paths = {
        str(path.relative_to(docs_dir))
        for path in docs_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in (".pdf", ".md", ".markdown", ".txt")
    }

    # Files that used to exist but are no longer in docs/: remove their
    # chunks and drop them from the manifest.
    for relative_path in list(manifest.keys()):
        if relative_path not in current_paths:
            chunk_ids = manifest[relative_path]["chunk_ids"]
            if chunk_ids:
                vector_store.delete(ids=chunk_ids)
            del manifest[relative_path]
            _save_manifest(manifest)
            logger.info("Removed: %s (%d chunk(s) deleted)", relative_path, len(chunk_ids))

    # New or changed files: (re-)embed. Unchanged files are skipped.
    for relative_path in sorted(current_paths):
        path = docs_dir / relative_path
        file_hash = _hash_file(path)
        existing = manifest.get(relative_path)

        if existing is not None and existing["hash"] == file_hash:
            logger.info("Unchanged, skipping: %s", relative_path)
            continue

        # Delete-then-insert: covers both "this file changed" (its old
        # chunks need replacing) and "a previous run crashed after writing
        # some chunks but before updating the manifest" (self-heals any
        # leftover partial chunks before writing fresh ones).
        if existing is not None:
            old_chunk_ids = existing["chunk_ids"]
            if old_chunk_ids:
                vector_store.delete(ids=old_chunk_ids)

        chunks = _load_and_split_file(path, relative_path)
        chunk_ids = [str(uuid.uuid4()) for _ in chunks]
        if chunks:
            vector_store.add_documents(documents=chunks, ids=chunk_ids)

        # Only now, after the chunks are confirmed written, record this
        # file as done - so a crash before this line just means the file
        # gets reprocessed (safely) on the next run.
        manifest[relative_path] = {"hash": file_hash, "chunk_ids": chunk_ids}
        _save_manifest(manifest)
        verb = "Embedded" if existing is None else "Re-embedded"
        logger.info("%s: %s (%d chunk(s))", verb, relative_path, len(chunks))

    logger.info("Ingestion complete.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Wipe the vector store and manifest, then re-embed everything from scratch.",
    )
    args = parser.parse_args()
    ingest(rebuild=args.rebuild)


if __name__ == "__main__":
    main()
