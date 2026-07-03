import json
from pathlib import Path

from causalens.observability import append_tool_call_log


def test_append_tool_call_log_writes_one_jsonl_record(
    tmp_path: Path,
):
    """
    验证一次工具调用会写入一行合法 JSON，
    并且只记录结果摘要，而不是完整结果内容。
    """
    log_path = tmp_path / "logs" / "tool_calls.jsonl"

    append_tool_call_log(
        question="不同专业的平均 score 分别是多少？",
        dataset_path="data/student_performance.csv",
        tool_name="run_readonly_sql",
        arguments={
            "dataset_path": "data/student_performance.csv",
            "query": (
                "SELECT major, AVG(score) AS avg_score "
                "FROM dataset GROUP BY major"
            ),
        },
        success=True,
        result={
            "row_count": 40,
            "returned_row_count": 3,
            "truncated": False,
            "columns": [
                "major",
                "avg_score",
            ],
            "rows": [
                {
                    "major": "cs",
                    "avg_score": 81.23,
                },
                {
                    "major": "math",
                    "avg_score": 79.57,
                },
                {
                    "major": "stats",
                    "avg_score": 74.69,
                },
            ],
        },
        error=None,
        duration_ms=12.3456,
        log_path=log_path,
    )

    assert log_path.exists()

    lines = log_path.read_text(
        encoding="utf-8",
    ).splitlines()

    assert len(lines) == 1

    record = json.loads(lines[0])

    assert record["question"] == (
        "不同专业的平均 score 分别是多少？"
    )
    assert record["dataset_path"] == (
        "data/student_performance.csv"
    )
    assert record["tool_name"] == "run_readonly_sql"
    assert record["success"] is True
    assert record["duration_ms"] == 12.35
    assert record["error"] is None

    summary = record["result_summary"]

    assert summary["row_count"] == 40
    assert summary["returned_row_count"] == 3
    assert summary["rows_returned"] == 3
    assert summary["truncated"] is False

    # 完整 rows 不应该被直接写入摘要。
    assert "rows" not in summary