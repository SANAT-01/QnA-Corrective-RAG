"""Load documents from docs/, split, embed, and persist to a local Chroma store.

v0.2: incremental, idempotent ingestion. A manifest file
(chroma_db/manifest.json) tracks, per file (keyed by its path relative to
docs/), the content hash and the Chroma chunk ids it produced. Each run:

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
import shutil
import uuid
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from qna.llm import get_embeddings

DOCS_DIR = Path("docs")
CHROMA_DIR = Path("chroma_db")
MANIFEST_PATH = CHROMA_DIR / "manifest.json"
COLLECTION_NAME = "qna_docs"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

MARKDOWN_HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3")]


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text())
    return {}


def _save_manifest(manifest: dict) -> None:
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))


def _load_and_split_file(path: Path, relative_path: str) -> list[Document]:
    """Load a single file and split it into chunks, tagging each chunk's
    metadata with its source path (relative to docs/) plus, where
    available, page number (PDFs) or nearest heading (markdown) - used for
    citations starting in v0.5.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )

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


def ingest(rebuild: bool = False) -> None:
    if rebuild and CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)

    if not DOCS_DIR.exists() or not any(DOCS_DIR.iterdir()):
        print(f"No documents found in {DOCS_DIR}/. Add some PDFs or markdown files first.")
        return

    manifest = _load_manifest()
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(CHROMA_DIR),
    )

    current_paths = {
        str(path.relative_to(DOCS_DIR))
        for path in DOCS_DIR.rglob("*")
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
            print(f"Removed: {relative_path} ({len(chunk_ids)} chunk(s) deleted)")

    # New or changed files: (re-)embed. Unchanged files are skipped.
    for relative_path in sorted(current_paths):
        path = DOCS_DIR / relative_path
        file_hash = _hash_file(path)
        existing = manifest.get(relative_path)

        if existing is not None and existing["hash"] == file_hash:
            print(f"Unchanged, skipping: {relative_path}")
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
        print(f"{verb}: {relative_path} ({len(chunks)} chunk(s))")

    print("Ingestion complete.")


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
