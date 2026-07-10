import pytest

from causalens.rag.retriever import format_chroma_query_results, validate_top_k


def test_validate_top_k_accepts_positive_integer() -> None:
    validate_top_k(1)
    validate_top_k(3)


def test_validate_top_k_rejects_zero() -> None:
    with pytest.raises(ValueError):
        validate_top_k(0)


def test_validate_top_k_rejects_negative_integer() -> None:
    with pytest.raises(ValueError):
        validate_top_k(-1)


def test_format_chroma_query_results_converts_nested_results() -> None:
    query_results = {
        "ids": [["data_dictionary_0001"]],
        "documents": [["passed means whether the student passed the examination."]],
        "metadatas": [
            [
                {
                    "chunk_id": "data_dictionary_0001",
                    "source": "data_dictionary.md",
                    "path": "knowledge/data_dictionary.md",
                    "chunk_index": 1,
                }
            ]
        ],
        "distances": [[0.123]],
    }

    results = format_chroma_query_results(query_results)

    assert results == [
        {
            "source": "data_dictionary.md",
            "chunk_id": "data_dictionary_0001",
            "chunk_index": 1,
            "path": "knowledge/data_dictionary.md",
            "content": "passed means whether the student passed the examination.",
            "distance": 0.123,
        }
    ]


def test_format_chroma_query_results_uses_id_as_fallback_chunk_id() -> None:
    query_results = {
        "ids": [["fallback_0001"]],
        "documents": [["Some retrieved content."]],
        "metadatas": [[{}]],
        "distances": [[0.5]],
    }

    results = format_chroma_query_results(query_results)

    assert results[0]["chunk_id"] == "fallback_0001"
    assert results[0]["source"] == ""
    assert results[0]["content"] == "Some retrieved content."
    assert results[0]["distance"] == 0.5


def test_format_chroma_query_results_preserves_section_metadata() -> None:
    query_results = {
        "ids": [["data_dictionary_0007"]],
        "documents": [["score means final examination score."]],
        "metadatas": [
            [
                {
                    "chunk_id": "data_dictionary_0007",
                    "source": "data_dictionary.md",
                    "path": "knowledge/data_dictionary.md",
                    "chunk_index": 7,
                    "section": "score",
                }
            ]
        ],
        "distances": [[0.2]],
    }

    results = format_chroma_query_results(query_results)

    assert results[0]["source"] == "data_dictionary.md"
    assert results[0]["section"] == "score"
    assert results[0]["chunk_id"] == "data_dictionary_0007"