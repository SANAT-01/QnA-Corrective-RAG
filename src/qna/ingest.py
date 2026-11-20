"""Load documents from docs/, split, embed, and persist to a local Chroma store.

v0.1: simplest possible version. Every run wipes and rebuilds the whole
vector store from whatever is currently in docs/. Incremental, resumable
ingestion comes in v0.2.
"""

import shutil
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from qna.llm import get_embeddings

DOCS_DIR = Path("docs")
CHROMA_DIR = Path("chroma_db")
COLLECTION_NAME = "qna_docs"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def load_documents():
    """Load every PDF/markdown/text file in docs/ into LangChain Document objects.

    Each loaded Document carries `metadata["source"]` set to its file path -
    that's what graph.py's `generate` node uses to cite sources in answers.
    """
    documents = []
    for path in sorted(DOCS_DIR.iterdir()):
        if path.suffix.lower() == ".pdf":
            documents.extend(PyPDFLoader(str(path)).load())
        elif path.suffix.lower() in (".md", ".markdown", ".txt"):
            documents.extend(TextLoader(str(path)).load())
    return documents


def main() -> None:
    if not DOCS_DIR.exists() or not any(DOCS_DIR.iterdir()):
        print(f"No documents found in {DOCS_DIR}/. Add some PDFs or markdown files first.")
        return

    documents = load_documents()
    print(f"Loaded {len(documents)} document(s) from {DOCS_DIR}/")

    # Splits each loaded document into ~CHUNK_SIZE-character pieces, with
    # CHUNK_OVERLAP characters of overlap between consecutive chunks so a
    # fact that straddles a chunk boundary still appears intact in at
    # least one chunk. These are the units that actually get embedded and
    # retrieved - not whole files.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunk(s)")

    # v0.1: full rebuild every run - simplest possible approach while we're
    # focused on learning the graph itself. This nukes any previously
    # ingested data unconditionally; v0.2 replaces this with a manifest
    # that only re-embeds new/changed files.
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)

    # This single call does three things: embeds every chunk's text via
    # get_embeddings() (one call to our self-hosted API per batch), creates
    # a fresh Chroma collection, and writes the chunks + their vectors +
    # metadata into CHROMA_DIR on disk.
    Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
    )
    print(f"Embedded and persisted {len(chunks)} chunk(s) to {CHROMA_DIR}/")


if __name__ == "__main__":
    main()
