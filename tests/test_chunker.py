import pytest

from causalens.rag.chunker import create_chunks_from_documents, split_text


def test_split_text_returns_single_chunk_for_short_text() -> None:
    chunks = split_text("hello world", chunk_size=100, chunk_overlap=20)

    assert chunks == ["hello world"]


def test_split_text_splits_long_text_with_overlap() -> None:
    text = "abcdefghij"

    chunks = split_text(text, chunk_size=4, chunk_overlap=1)

    assert chunks == ["abcd", "defg", "ghij", "j"]


def test_split_text_returns_empty_list_for_empty_text() -> None:
    chunks = split_text("", chunk_size=100, chunk_overlap=20)

    assert chunks == []


def test_split_text_rejects_invalid_chunk_size() -> None:
    with pytest.raises(ValueError):
        split_text("hello", chunk_size=0, chunk_overlap=0)


def test_split_text_rejects_negative_overlap() -> None:
    with pytest.raises(ValueError):
        split_text("hello", chunk_size=100, chunk_overlap=-1)


def test_split_text_rejects_overlap_not_smaller_than_chunk_size() -> None:
    with pytest.raises(ValueError):
        split_text("hello", chunk_size=100, chunk_overlap=100)


def test_create_chunks_from_documents_adds_metadata() -> None:
    documents = [
        {
            "source": "data_dictionary.md",
            "path": "knowledge/data_dictionary.md",
            "content": "abcdefghij",
        }
    ]

    chunks = create_chunks_from_documents(
        documents=documents,
        chunk_size=4,
        chunk_overlap=1,
    )

    assert len(chunks) == 4

    assert chunks[0]["chunk_id"] == "data_dictionary_0000"
    assert chunks[0]["source"] == "data_dictionary.md"
    assert chunks[0]["path"] == "knowledge/data_dictionary.md"
    assert chunks[0]["content"] == "abcd"
    assert chunks[0]["chunk_index"] == 0

    assert chunks[1]["chunk_id"] == "data_dictionary_0001"
    assert chunks[1]["content"] == "defg"
    assert chunks[1]["chunk_index"] == 1