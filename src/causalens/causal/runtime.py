from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any
from uuid import uuid4


CACHE_SCHEMA_VERSION = 1
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def dataset_fingerprint(dataset_path: str | Path) -> dict[str, Any]:
    """计算内容哈希，避免路径相同但数据变化时复用陈旧结果。"""

    path = Path(dataset_path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    stat = path.stat()
    return {
        "sha256": digest.hexdigest(),
        "size_bytes": stat.st_size,
        "modified_ns": stat.st_mtime_ns,
    }


def build_analysis_cache_key(
    *,
    dataset_path: str | Path,
    request: dict[str, Any],
) -> str:
    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "dataset": dataset_fingerprint(dataset_path),
        "request": request,
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _atomic_json_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    temporary.replace(path)


class AnalysisResultCache:
    """内容寻址、原子写入的本地成功结果缓存。"""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory).resolve()

    def load(self, key: str) -> dict[str, Any] | None:
        if not re.fullmatch(r"[0-9a-f]{64}", key):
            raise ValueError("Invalid analysis cache key.")
        path = self.directory / f"{key}.json"
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if payload.get("schema_version") != CACHE_SCHEMA_VERSION:
            return None
        value = payload.get("value")
        return value if isinstance(value, dict) else None

    def store(self, key: str, value: dict[str, Any]) -> Path:
        if not re.fullmatch(r"[0-9a-f]{64}", key):
            raise ValueError("Invalid analysis cache key.")
        path = self.directory / f"{key}.json"
        _atomic_json_write(
            path,
            {
                "schema_version": CACHE_SCHEMA_VERSION,
                "key": key,
                "value": value,
            },
        )
        return path


class WorkflowCheckpointStore:
    """保存可恢复的本地 Agent 状态快照，不包含任何 API Key。"""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory).resolve()

    def _path(self, run_id: str) -> Path:
        if not _SAFE_ID.fullmatch(run_id):
            raise ValueError(
                "run_id must contain only letters, numbers, dots, underscores, "
                "or hyphens and be at most 128 characters."
            )
        return self.directory / f"{run_id}.json"

    def save(self, run_id: str, state: dict[str, Any]) -> Path:
        path = self._path(run_id)
        _atomic_json_write(
            path,
            {
                "schema_version": CACHE_SCHEMA_VERSION,
                "run_id": run_id,
                "state": state,
            },
        )
        return path

    def load(self, run_id: str) -> dict[str, Any] | None:
        path = self._path(run_id)
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if payload.get("schema_version") != CACHE_SCHEMA_VERSION:
            return None
        state = payload.get("state")
        return state if isinstance(state, dict) else None
