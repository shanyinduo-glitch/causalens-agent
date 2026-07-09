from __future__ import annotations

from causalens.rag.retriever import retrieve_documents
from causalens.schemas import ToolResult


def retrieve_docs(query: str, top_k: int = 3) -> ToolResult:
    """
    Tool wrapper for retrieving project documentation.

    This tool is intended for questions about:
    - column meanings
    - units
    - dataset background
    - data limitations
    - analysis assumptions
    - business rules
    """
    try:
        results = retrieve_documents(
            query=query,
            top_k=top_k,
        )

        return ToolResult(
            tool_name="retrieve_docs",
            success=True,
            result={
                "query": query,
                "results": results,
            },
        )

    except Exception as exc:
        return ToolResult(
            tool_name="retrieve_docs",
            success=False,
            error=str(exc),
        )