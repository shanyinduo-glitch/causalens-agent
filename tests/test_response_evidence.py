from causalens.response_evidence import build_evidence_rows
from causalens.schemas import ToolResult


def test_build_evidence_rows_for_successful_tools():
    tool_results = [
        ToolResult(
            tool_name="profile_dataset",
            success=True,
            result={
                "rows": 40,
                "columns": 8,
                "column_names": [
                    "student_id",
                    "major",
                    "city",
                    "study_hours",
                    "attendance_rate",
                    "score",
                    "passed",
                    "exam_date",
                ],
            },
        ),
        ToolResult(
            tool_name="run_readonly_sql",
            success=True,
            result={
                "returned_row_count": 3,
                "columns": [
                    "major",
                    "avg_score",
                    "student_count",
                ],
                "truncated": False,
            },
        ),
        ToolResult(
            tool_name="create_chart",
            success=True,
            result={
                "chart_path": (
                    "outputs/charts/"
                    "study_hours_vs_score_test.png"
                ),
                "chart_type": "scatter",
                "plotted_row_count": 40,
                "relationship_summary": {
                    "point_count": 40,
                    "x_min": 3.0,
                    "x_max": 18.0,
                    "y_min": 52.0,
                    "y_max": 98.0,
                    "pearson_correlation": 0.9874,
                },
            },
        ),
    ]

    rows = build_evidence_rows(tool_results)

    assert len(rows) == 3

    assert rows[0]["tool_name"] == "profile_dataset"
    assert rows[0]["status"] == "成功"
    assert "40 行 × 8 列" in rows[0]["evidence"]
    assert "student_id、major、city" in rows[0]["evidence"]

    assert rows[1]["tool_name"] == "run_readonly_sql"
    assert rows[1]["status"] == "成功"
    assert "SQL 返回 3 行" in rows[1]["evidence"]
    assert "major、avg_score、student_count" in rows[1]["evidence"]

    assert rows[2]["tool_name"] == "create_chart"
    assert rows[2]["status"] == "成功"
    assert "生成 scatter 图" in rows[2]["evidence"]
    assert "参与绘图数据：40 条" in rows[2]["evidence"]
    assert "pearson_correlation=0.9874" in rows[2]["evidence"]


def test_build_evidence_rows_for_failed_tool():
    tool_results = [
        ToolResult(
            tool_name="create_chart",
            success=False,
            error="无法生成散点图：x_column=major 必须是数值列。",
        ),
    ]

    rows = build_evidence_rows(tool_results)

    assert rows == [
        {
            "tool_name": "create_chart",
            "status": "失败",
            "evidence": (
                "错误：无法生成散点图："
                "x_column=major 必须是数值列。"
            ),
        }
    ]

def test_build_evidence_rows_formats_retrieve_docs_result() -> None:
    tool_results = [
        ToolResult(
            tool_name="retrieve_docs",
            success=True,
            result={
                "query": "passed 字段是什么意思？",
                "results": [
                    {
                        "source": "data_dictionary.md",
                        "chunk_id": "data_dictionary_0008",
                        "distance": 0.1234,
                        "content": "1 means passed. 0 means failed.",
                    }
                ],
            },
        )
    ]

    rows = build_evidence_rows(tool_results)

    assert len(rows) == 1
    assert rows[0]["tool_name"] == "retrieve_docs"
    assert rows[0]["status"] == "成功"
    assert "检索问题：passed 字段是什么意思？" in rows[0]["evidence"]
    assert "返回片段数：1" in rows[0]["evidence"]
    assert "data_dictionary.md" in rows[0]["evidence"]
    assert "data_dictionary_0008" in rows[0]["evidence"]