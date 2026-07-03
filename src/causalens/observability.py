from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


TOOL_CALL_LOG_PATH = Path("logs/tool_calls.jsonl")


def _summarize_result(
    result: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """
    将工具结果压缩成适合记录日志的摘要。

    不直接完整记录所有结果行，避免 SQL 查询结果过大、
    日志膨胀，或意外写入过多数据内容。
    """
    if result is None:
        return None

    summary: dict[str, Any] = {
        "result_keys": sorted(result.keys()),
    }

    important_keys = [
        "row_count",
        "returned_row_count",
        "truncated",
        "chart_path",
        "chart_type",
        "value_column",
        "plotted_row_count",
    ]

    for key in important_keys:
        if key in result:
            summary[key] = result[key]

    rows = result.get("rows")
    if isinstance(rows, list):
        summary["rows_returned"] = len(rows)

    data_preview = result.get("data_preview")
    if isinstance(data_preview, list):
        summary["data_preview_row_count"] = len(data_preview)

    return summary


def append_tool_call_log(
    *,
    question: str,
    dataset_path: str,
    tool_name: str,
    arguments: dict[str, Any],
    success: bool,
    result: dict[str, Any] | None,
    error: str | None,
    duration_ms: float,
    log_path: Path | None = None,
) -> None:
    """
    将一次工具调用追加写入 JSONL 日志文件。

    JSONL 的含义是“一行一个 JSON 对象”，适合持续追加、
    后续筛选和批量分析。
    """
    target_path = log_path or TOOL_CALL_LOG_PATH

    record = {
        "timestamp": datetime.now().astimezone().isoformat(
            timespec="seconds"
        ),
        "question": question,
        "dataset_path": dataset_path,
        "tool_name": tool_name,
        "arguments": arguments,
        "success": success,
        "duration_ms": round(duration_ms, 2),
        "result_summary": _summarize_result(result),
        "error": error,
    }

    target_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with target_path.open(
        "a",
        encoding="utf-8",
    ) as log_file:
        log_file.write(
            json.dumps(
                record,
                ensure_ascii=False,
                default=str,
            )
            + "\n"
        )