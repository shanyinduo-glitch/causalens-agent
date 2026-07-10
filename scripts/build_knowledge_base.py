from __future__ import annotations

from pathlib import Path

from causalens.rag.chunker import create_chunks_from_documents
from causalens.rag.document_loader import load_markdown_documents
from causalens.rag.vector_store import (
    DEFAULT_COLLECTION_NAME,
    DEFAULT_EMBEDDING_MODEL_NAME,
    DEFAULT_PERSIST_DIR,
    build_vector_store,
)


def main() -> None:
    knowledge_dir = Path("knowledge")

    print("=" * 60)
    print("Building CausaLens Knowledge Base")
    print("=" * 60)

    documents = load_markdown_documents(knowledge_dir)

    chunks = create_chunks_from_documents(
        documents=documents,
        chunk_size=900,
        chunk_overlap=120,
        chunk_strategy="markdown_heading",
    )

    summary = build_vector_store(
        chunks=chunks,
        persist_dir=DEFAULT_PERSIST_DIR,
        collection_name=DEFAULT_COLLECTION_NAME,
        model_name=DEFAULT_EMBEDDING_MODEL_NAME,
        reset_collection=True,
    )

    print(f"Loaded {len(documents)} documents.")
    print(f"Created {len(chunks)} text chunks.")
    print(f"Stored {summary['stored_count']} vectors in local vector database.")
    print(f"Persist directory: {summary['persist_dir']}")
    print(f"Collection name: {summary['collection_name']}")
    print(f"Embedding model: {summary['embedding_model']}")
    print(f"Embedding dimension: {summary['embedding_dimension']}")


if __name__ == "__main__":
    main()