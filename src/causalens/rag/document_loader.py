from __future__ import annotations

from pathlib import Path
from typing import Any


def load_markdown_documents(folder_path: str | Path) -> list[dict[str, Any]]:
    """
    Load all Markdown documents from a folder.

    Each returned document keeps:
    - source: file name, such as data_dictionary.md
    - path: full file path as string
    - content: raw Markdown text

    Parameters
    ----------
    folder_path:
        Folder containing Markdown files.

    Returns
    -------
    list[dict[str, Any]]
        A list of loaded Markdown documents.
    """
    folder = Path(folder_path)

    if not folder.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder}")

    if not folder.is_dir():
        raise NotADirectoryError(f"Path is not a folder: {folder}")

    documents: list[dict[str, Any]] = []

    for file_path in sorted(folder.glob("*.md")):
        content = file_path.read_text(encoding="utf-8")

        documents.append(
            {
                "source": file_path.name,
                "path": str(file_path),
                "content": content,
            }
        )

    return documents