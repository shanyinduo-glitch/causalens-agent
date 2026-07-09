from __future__ import annotations

from pathlib import Path
from typing import Any


DEFAULT_COLLECTION_NAME = "causalens_knowledge"
DEFAULT_EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_PERSIST_DIR = "storage/chroma_db"


def embed_texts(
    texts: list[str],
    model_name: str = DEFAULT_EMBEDDING_MODEL_NAME,
) -> list[list[float]]:
    if not texts:
        return []

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    return embeddings.tolist()


def prepare_chroma_records(
    chunks: list[dict[str, Any]],
    embeddings: list[list[float]],
) -> dict[str, Any]:
    if len(chunks) != len(embeddings):
        raise ValueError(
            f"chunks and embeddings must have the same length: "
            f"{len(chunks)} != {len(embeddings)}"
        )

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict[str, Any]] = []

    for chunk in chunks:
        chunk_id = str(chunk["chunk_id"])

        ids.append(chunk_id)
        documents.append(str(chunk["content"]))
        metadatas.append(
            {
                "chunk_id": chunk_id,
                "source": str(chunk["source"]),
                "path": str(chunk.get("path", "")),
                "chunk_index": int(chunk["chunk_index"]),
            }
        )

    return {
        "ids": ids,
        "documents": documents,
        "metadatas": metadatas,
        "embeddings": embeddings,
    }


def get_persistent_client(persist_dir: str | Path = DEFAULT_PERSIST_DIR) -> Any:
    import chromadb

    persist_path = Path(persist_dir)
    persist_path.mkdir(parents=True, exist_ok=True)

    return chromadb.PersistentClient(path=str(persist_path))


def delete_collection_if_exists(client: Any, collection_name: str) -> None:
    try:
        client.delete_collection(name=collection_name)
    except Exception as exc:
        message = str(exc).lower()

        expected_missing_messages = [
            "does not exist",
            "not found",
            "doesn't exist",
        ]

        if not any(text in message for text in expected_missing_messages):
            raise


def build_vector_store(
    chunks: list[dict[str, Any]],
    persist_dir: str | Path = DEFAULT_PERSIST_DIR,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    model_name: str = DEFAULT_EMBEDDING_MODEL_NAME,
    reset_collection: bool = True,
) -> dict[str, Any]:
    if not chunks:
        raise ValueError("Cannot build vector store from empty chunks")

    texts = [str(chunk["content"]) for chunk in chunks]
    embeddings = embed_texts(texts=texts, model_name=model_name)

    records = prepare_chroma_records(chunks=chunks, embeddings=embeddings)

    client = get_persistent_client(persist_dir=persist_dir)

    if reset_collection:
        delete_collection_if_exists(client=client, collection_name=collection_name)

    collection = client.get_or_create_collection(name=collection_name)

    collection.add(
        ids=records["ids"],
        documents=records["documents"],
        metadatas=records["metadatas"],
        embeddings=records["embeddings"],
    )

    embedding_dimension = len(records["embeddings"][0]) if records["embeddings"] else 0

    return {
        "persist_dir": str(persist_dir),
        "collection_name": collection_name,
        "stored_count": collection.count(),
        "embedding_model": model_name,
        "embedding_dimension": embedding_dimension,
    }