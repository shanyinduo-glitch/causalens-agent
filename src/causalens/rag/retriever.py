from __future__ import annotations

from pathlib import Path
from typing import Any

from causalens.rag.vector_store import (
    DEFAULT_COLLECTION_NAME,
    DEFAULT_EMBEDDING_MODEL_NAME,
    DEFAULT_PERSIST_DIR,
    embed_texts,
    get_persistent_client,
)


def validate_top_k(top_k: int) -> None:
    """
    Validate the number of retrieved chunks.
    """
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")


def format_chroma_query_results(query_results: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Convert Chroma query results into a simple list of retrieval results.

    Chroma returns nested lists because it supports batch queries.
    In this project, we query one question at a time, so we read index 0.
    """
    ids_batch = query_results.get("ids") or [[]]
    documents_batch = query_results.get("documents") or [[]]
    metadatas_batch = query_results.get("metadatas") or [[]]
    distances_batch = query_results.get("distances") or [[]]

    ids = ids_batch[0] if ids_batch else []
    documents = documents_batch[0] if documents_batch else []
    metadatas = metadatas_batch[0] if metadatas_batch else []
    distances = distances_batch[0] if distances_batch else []

    results: list[dict[str, Any]] = []

    for index, content in enumerate(documents):
        metadata = metadatas[index] if index < len(metadatas) and metadatas[index] else {}
        distance = distances[index] if index < len(distances) else None
        fallback_id = ids[index] if index < len(ids) else ""

        chunk_id = str(metadata.get("chunk_id") or fallback_id)
        source = str(metadata.get("source", ""))
        path = str(metadata.get("path", ""))
        chunk_index = metadata.get("chunk_index")

        result_item = {
            "source": source,
            "chunk_id": chunk_id,
            "chunk_index": chunk_index,
            "path": path,
            "content": content,
            "distance": distance,
        }

        section = metadata.get("section")
        if section:
            result_item["section"] = section

        results.append(result_item)

    return results


def retrieve_documents(
    query: str,
    top_k: int = 3,
    persist_dir: str | Path = DEFAULT_PERSIST_DIR,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    model_name: str = DEFAULT_EMBEDDING_MODEL_NAME,
) -> list[dict[str, Any]]:
    """
    Retrieve the most relevant document chunks from the local Chroma vector store.

    Parameters
    ----------
    query:
        User question or search query.
    top_k:
        Number of chunks to retrieve.
    persist_dir:
        Local Chroma database directory.
    collection_name:
        Chroma collection name.
    model_name:
        Sentence Transformers model name.

    Returns
    -------
    list[dict[str, Any]]
        Retrieved chunks with source metadata.
    """
    validate_top_k(top_k)

    if not query.strip():
        raise ValueError("query cannot be empty")

    client = get_persistent_client(persist_dir=persist_dir)
    collection = client.get_collection(name=collection_name)

    query_embedding = embed_texts(
        texts=[query],
        model_name=model_name,
    )[0]

    query_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    return format_chroma_query_results(query_results)