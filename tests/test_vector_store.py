import pytest

from causalens.rag.vector_store import prepare_chroma_records


def test_prepare_chroma_records_converts_chunks_and_embeddings() -> None:
    chunks = [
        {
            "chunk_id": "data_dictionary_0000",
            "source": "data_dictionary.md",
            "path": "knowledge/data_dictionary.md",
            "content": "passed means whether the student passed the exam.",
            "chunk_index": 0,
        },
        {
            "chunk_id": "analysis_rules_0000",
            "source": "analysis_rules.md",
            "path": "knowledge/analysis_rules.md",
            "content": "Correlation does not establish causality.",
            "chunk_index": 0,
        },
    ]

    embeddings = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
    ]

    records = prepare_chroma_records(chunks=chunks, embeddings=embeddings)

    assert records["ids"] == [
        "data_dictionary_0000",
        "analysis_rules_0000",
    ]

    assert records["documents"] == [
        "passed means whether the student passed the exam.",
        "Correlation does not establish causality.",
    ]

    assert records["metadatas"] == [
        {
            "chunk_id": "data_dictionary_0000",
            "source": "data_dictionary.md",
            "path": "knowledge/data_dictionary.md",
            "chunk_index": 0,
        },
        {
            "chunk_id": "analysis_rules_0000",
            "source": "analysis_rules.md",
            "path": "knowledge/analysis_rules.md",
            "chunk_index": 0,
        },
    ]

    assert records["embeddings"] == embeddings


def test_prepare_chroma_records_rejects_length_mismatch() -> None:
    chunks = [
        {
            "chunk_id": "data_dictionary_0000",
            "source": "data_dictionary.md",
            "path": "knowledge/data_dictionary.md",
            "content": "hello",
            "chunk_index": 0,
        }
    ]

    embeddings = []

    with pytest.raises(ValueError):
        prepare_chroma_records(chunks=chunks, embeddings=embeddings)