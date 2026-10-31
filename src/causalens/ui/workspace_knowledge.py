"""Dataset-scoped knowledge bases created from user-provided research context."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
from typing import Any

from causalens.rag.chunker import create_chunks_from_documents
from causalens.rag.retriever import retrieve_documents
from causalens.rag.vector_store import (
    DEFAULT_EMBEDDING_MODEL_NAME,
    DEFAULT_PERSIST_DIR,
    build_vector_store,
    get_persistent_client,
)


def _knowledge_root(root: str | Path) -> Path:
    return Path(root) / "storage" / "workspace_knowledge"


def _manifest_path(root: str | Path, dataset_sha256: str) -> Path:
    return _knowledge_root(root) / dataset_sha256 / "manifest.json"


def _records_path(root: str | Path, dataset_sha256: str) -> Path:
    return _knowledge_root(root) / dataset_sha256 / "records.json"


def collection_name(dataset_sha256: str) -> str:
    """Return a stable, Chroma-safe collection name for one dataset version."""

    normalized = "".join(ch for ch in dataset_sha256.lower() if ch.isalnum())
    if len(normalized) < 8:
        raise ValueError("dataset_sha256 must contain at least 8 hexadecimal characters")
    return f"workspace_{normalized[:32]}"


def load_workspace_manifest(root: str | Path, dataset_sha256: str) -> dict[str, Any] | None:
    path = _manifest_path(root, dataset_sha256)
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def load_workspace_records(root: str | Path, dataset_sha256: str) -> list[dict[str, Any]]:
    path = _records_path(root, dataset_sha256)
    if not path.is_file():
        return []
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return value if isinstance(value, list) else []


def build_workspace_knowledge(
    root: str | Path,
    dataset_sha256: str,
    background: str,
    *,
    model_name: str = DEFAULT_EMBEDDING_MODEL_NAME,
    dataset_context: str = "",
) -> dict[str, Any]:
    """Persist a dataset-scoped local knowledge base.

    The corpus contains the user's research background and, when supplied, a
    generated schema/data-context document. The source text remains local
    under ``storage/``. It is never sent to an external model by this function;
    the normal workspace consent still gates any later LLM request that uses
    retrieved excerpts.
    """

    text = str(background or "").strip()
    if len(dataset_sha256) < 8:
        raise ValueError("数据版本标识无效，无法隔离知识库。")

    source = "user_research_background.md"
    context = str(dataset_context or "").strip()
    if len(text) < 20 and not context:
        raise ValueError("请填写至少 20 个字符的研究背景，或提供数据结构摘要。")
    documents = []
    if text:
        documents.append({"source": source, "path": source, "content": text})
    if context:
        documents.append({
            "source": "dataset_schema_and_profile.md",
            "path": "dataset_schema_and_profile.md",
            "content": context,
        })
    chunks = create_chunks_from_documents(
        documents=documents,
        chunk_size=900,
        chunk_overlap=120,
        chunk_strategy="markdown_heading",
    )
    if not chunks:
        raise ValueError("研究背景没有可建立的文本片段。")

    for chunk in chunks:
        chunk["chunk_id"] = (
            f"{collection_name(dataset_sha256)}_{chunk['chunk_index']:04d}"
        )

    persist_dir = Path(root) / DEFAULT_PERSIST_DIR
    vector_error = None
    try:
        vector_summary = build_vector_store(
            chunks=chunks,
            persist_dir=persist_dir,
            collection_name=collection_name(dataset_sha256),
            model_name=model_name,
            reset_collection=True,
        )
        storage_mode = "vector"
    except Exception as exc:
        # The source records are still a valid local knowledge base when an
        # embedding model is not cached or the environment has no network.
        # Retrieval will use the deterministic lexical fallback below.
        vector_error = f"{type(exc).__name__}: {exc}"
        vector_summary = {"stored_count": 0, "error": vector_error}
        storage_mode = "lexical_fallback"

    record_time = datetime.now(timezone.utc).isoformat()
    records = []
    for chunk in chunks:
        source_label = (
            "用户研究背景"
            if chunk.get("source") == source
            else "当前数据集结构与概况"
        )
        records.append({
            "chunk_id": chunk["chunk_id"],
            "source": source_label,
            "source_file": chunk.get("source", ""),
            "section": chunk.get("section", "Document"),
            "text": chunk["content"],
            "chunk_index": chunk["chunk_index"],
        })
    manifest = {
        "dataset_sha256": dataset_sha256,
        "background_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "collection_name": collection_name(dataset_sha256),
        "embedding_model": model_name,
        "chunk_count": len(chunks),
        "document_count": len(documents),
        "source_files": [str(document["source"]) for document in documents],
        "chunking": {"strategy": "markdown_heading", "chunk_size": 900, "chunk_overlap": 120},
        "created_at": record_time,
        "updated_at": record_time,
        "storage_scope": "local_dataset_version",
        "storage_mode": storage_mode,
        "vector_summary": vector_summary,
    }
    if vector_error:
        manifest["vector_error"] = vector_error[:500]
    target = _knowledge_root(root) / dataset_sha256
    target.mkdir(parents=True, exist_ok=True)
    _records_path(root, dataset_sha256).write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _manifest_path(root, dataset_sha256).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def delete_workspace_knowledge(root: str | Path, dataset_sha256: str) -> bool:
    """Delete the local KB for exactly one dataset version."""

    manifest = load_workspace_manifest(root, dataset_sha256) or {}
    collection = manifest.get("collection_name")
    if collection:
        try:
            client = get_persistent_client(Path(root) / DEFAULT_PERSIST_DIR)
            client.delete_collection(name=str(collection))
        except Exception as exc:
            # A missing Chroma install or collection should not prevent removal
            # of the source records and manifest.
            message = str(exc).lower()
            if not any(marker in message for marker in ("not found", "does not exist", "doesn't exist")):
                pass
    target = _knowledge_root(root) / dataset_sha256
    if not target.exists():
        return False
    shutil.rmtree(target)
    return True


def _lexical_terms(value: str) -> set[str]:
    """Tokenize Latin words and Chinese characters/bigrams for offline recall."""

    terms: set[str] = set()
    for piece in re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]+", value.lower()):
        if re.fullmatch(r"[\u4e00-\u9fff]+", piece):
            terms.update(piece)
            terms.update(piece[index:index + 2] for index in range(len(piece) - 1))
        else:
            terms.add(piece)
    return {term for term in terms if term.strip()}


def retrieve_workspace_knowledge(
    root: str | Path,
    dataset_sha256: str,
    query: str,
    *,
    top_k: int = 4,
) -> list[dict[str, Any]]:
    """Retrieve cited excerpts from the current dataset's local knowledge base."""

    if not str(query or "").strip() or not load_workspace_manifest(root, dataset_sha256):
        return []
    manifest = load_workspace_manifest(root, dataset_sha256) or {}
    results = []
    # Do not retry a known offline build on every question. The manifest is
    # the source of truth for the active retrieval mode.
    if manifest.get("storage_mode", "vector") == "vector":
        try:
            results = retrieve_documents(
                query=query,
                top_k=top_k,
                persist_dir=Path(root) / DEFAULT_PERSIST_DIR,
                collection_name=str(manifest["collection_name"]),
                model_name=str(manifest.get("embedding_model") or DEFAULT_EMBEDDING_MODEL_NAME),
            )
        except Exception:
            results = []

    if not results:
        # Deterministic fallback for offline machines or an uncached embedding
        # model. It is intentionally lexical and clearly labeled in the UI.
        query_lower = str(query).lower()
        query_terms = _lexical_terms(query_lower)
        scored = []
        for record in load_workspace_records(root, dataset_sha256):
            text = str(record.get("text") or "")
            text_lower = text.lower()
            text_terms = _lexical_terms(text_lower)
            score = sum(1 for term in query_terms if term in text_terms)
            if query_lower.strip() and query_lower.strip() in text_lower:
                score += 3
            if score:
                scored.append((score, record))
        scored.sort(key=lambda item: (-item[0], item[1].get("chunk_index", 0)))
        return [
            {
                "id": "K-" + str(record.get("chunk_id") or "")[-12:],
                "source": f"{record.get('source') or '本地知识库'}（词项检索）",
                "section": record.get("section") or "研究背景",
                "text": str(record.get("text") or ""),
                "score": score,
            }
            for score, record in scored[:top_k]
        ]

    output = []
    for item in results:
        content = str(item.get("content") or "").strip()
        if not content:
            continue
        output.append(
            {
                "id": "K-" + str(item.get("chunk_id") or "")[-12:],
                "source": str(item.get("source") or "用户建立的本地知识库"),
                "section": item.get("section") or "研究背景",
                "text": content,
                "distance": item.get("distance"),
            }
        )
    return output
