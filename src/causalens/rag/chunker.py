from __future__ import annotations

from pathlib import Path
from typing import Any


def split_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 80,
) -> list[str]:
    """
    Split text into overlapping chunks.

    This is the first simple chunking strategy:
    fixed-size character chunks with overlap.

    Parameters
    ----------
    text:
        Text to split.
    chunk_size:
        Maximum number of characters in each chunk.
    chunk_overlap:
        Number of characters shared between neighboring chunks.

    Returns
    -------
    list[str]
        Text chunks.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    if not text:
        return []

    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        if chunk:
            chunks.append(chunk)

        start += chunk_size - chunk_overlap

    return chunks


def create_chunks_from_documents(
    documents: list[dict[str, Any]],
    chunk_size: int = 500,
    chunk_overlap: int = 80,
) -> list[dict[str, Any]]:
    """
    Create metadata-rich chunks from loaded documents.

    Each chunk keeps:
    - chunk_id
    - source
    - path
    - content
    - chunk_index

    Parameters
    ----------
    documents:
        Documents returned by load_markdown_documents.
    chunk_size:
        Maximum number of characters in each chunk.
    chunk_overlap:
        Number of overlapping characters between neighboring chunks.

    Returns
    -------
    list[dict[str, Any]]
        Metadata-rich chunks.
    """
    all_chunks: list[dict[str, Any]] = []

    for document in documents:
        source = document["source"]
        path = document.get("path", "")
        content = document["content"]

        text_chunks = split_text(
            text=content,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        source_stem = Path(source).stem

        for chunk_index, chunk_content in enumerate(text_chunks):
            chunk_id = f"{source_stem}_{chunk_index:04d}"

            all_chunks.append(
                {
                    "chunk_id": chunk_id,
                    "source": source,
                    "path": path,
                    "content": chunk_content,
                    "chunk_index": chunk_index,
                }
            )

    return all_chunks