from pathlib import Path

import pytest

from causalens.rag.document_loader import load_markdown_documents


def test_load_markdown_documents_loads_md_files(tmp_path: Path) -> None:
    docs_dir = tmp_path / "knowledge"
    docs_dir.mkdir()

    (docs_dir / "data_dictionary.md").write_text("# Data Dictionary", encoding="utf-8")
    (docs_dir / "project_background.md").write_text("# Project Background", encoding="utf-8")
    (docs_dir / "notes.txt").write_text("This should be ignored.", encoding="utf-8")

    documents = load_markdown_documents(docs_dir)

    assert len(documents) == 2
    assert documents[0]["source"] == "data_dictionary.md"
    assert documents[1]["source"] == "project_background.md"
    assert documents[0]["content"] == "# Data Dictionary"
    assert documents[1]["content"] == "# Project Background"


def test_load_markdown_documents_returns_empty_list_when_no_md_files(tmp_path: Path) -> None:
    docs_dir = tmp_path / "knowledge"
    docs_dir.mkdir()

    (docs_dir / "notes.txt").write_text("No Markdown here.", encoding="utf-8")

    documents = load_markdown_documents(docs_dir)

    assert documents == []


def test_load_markdown_documents_raises_for_missing_folder(tmp_path: Path) -> None:
    missing_dir = tmp_path / "missing"

    with pytest.raises(FileNotFoundError):
        load_markdown_documents(missing_dir)


def test_load_markdown_documents_raises_for_file_path(tmp_path: Path) -> None:
    file_path = tmp_path / "not_a_folder.md"
    file_path.write_text("# Not a folder", encoding="utf-8")

    with pytest.raises(NotADirectoryError):
        load_markdown_documents(file_path)