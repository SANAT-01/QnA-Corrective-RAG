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

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunk(s)")

    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)

    Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
    )
    print(f"Embedded and persisted {len(chunks)} chunk(s) to {CHROMA_DIR}/")


if __name__ == "__main__":
    main()
