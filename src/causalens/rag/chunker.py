from __future__ import annotations

from pathlib import Path
from typing import Any


def split_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 80,
) -> list[str]:
    """
    Split text into overlapping fixed-size chunks.

    This is the first simple chunking strategy:
    fixed-size character chunks with overlap.
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


def _clean_markdown_heading(line: str) -> str:
    """
    Convert a Markdown heading line into a clean section title.

    Example:
    '### score' -> 'score'
    '## Dataset Overview' -> 'Dataset Overview'
    """
    return line.strip().lstrip("#").strip()


def split_markdown_by_headings(
    text: str,
    chunk_size: int = 900,
    chunk_overlap: int = 120,
) -> list[dict[str, str]]:
    """
    Split Markdown text by headings.

    This strategy tries to keep one Markdown section together.
    For example, the content under '### score' should stay in the same chunk
    instead of being mixed with the next field section.

    If a section is still too long, it will be further split by fixed-size chunks.

    Returns
    -------
    list[dict[str, str]]
        Each item contains:
        - content: chunk text
        - section: Markdown section title
    """
    if not text:
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    sections: list[dict[str, str]] = []
    current_section = "Document"
    current_lines: list[str] = []

    for line in text.splitlines(keepends=True):
        stripped_line = line.strip()

        if stripped_line.startswith("#"):
            if current_lines:
                sections.append(
                    {
                        "section": current_section,
                        "content": "".join(current_lines).strip(),
                    }
                )

            current_section = _clean_markdown_heading(stripped_line)
            current_lines = [line]
        else:
            current_lines.append(line)

    if current_lines:
        sections.append(
            {
                "section": current_section,
                "content": "".join(current_lines).strip(),
            }
        )

    chunks: list[dict[str, str]] = []

    for section in sections:
        section_title = section["section"]
        section_content = section["content"]

        if not section_content:
            continue

        if len(section_content) <= chunk_size:
            chunks.append(
                {
                    "section": section_title,
                    "content": section_content,
                }
            )
            continue

        sub_chunks = split_text(
            text=section_content,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        for sub_chunk in sub_chunks:
            chunks.append(
                {
                    "section": section_title,
                    "content": sub_chunk,
                }
            )

    return chunks


def create_chunks_from_documents(
    documents: list[dict[str, Any]],
    chunk_size: int = 500,
    chunk_overlap: int = 80,
    chunk_strategy: str = "fixed",
) -> list[dict[str, Any]]:
    """
    Create metadata-rich chunks from loaded documents.

    Parameters
    ----------
    documents:
        Documents returned by load_markdown_documents.
    chunk_size:
        Maximum number of characters in each chunk.
    chunk_overlap:
        Number of overlapping characters between neighboring chunks.
    chunk_strategy:
        'fixed' for fixed-size character chunks.
        'markdown_heading' for Markdown heading-aware chunks.

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

        if chunk_strategy == "fixed":
            text_chunks = [
                {
                    "content": chunk_content,
                    "section": "",
                }
                for chunk_content in split_text(
                    text=content,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
            ]

        elif chunk_strategy == "markdown_heading":
            text_chunks = split_markdown_by_headings(
                text=content,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )

        else:
            raise ValueError(
                "chunk_strategy must be either 'fixed' or 'markdown_heading'"
            )

        source_stem = Path(source).stem

        for chunk_index, chunk_data in enumerate(text_chunks):
            chunk_id = f"{source_stem}_{chunk_index:04d}"

            all_chunks.append(
                {
                    "chunk_id": chunk_id,
                    "source": source,
                    "path": path,
                    "content": chunk_data["content"],
                    "chunk_index": chunk_index,
                    "section": chunk_data.get("section", ""),
                }
            )

    return all_chunks