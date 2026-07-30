from datetime import datetime, timezone
from pathlib import Path

from causalens.reporting.markdown_report import (
    generate_markdown_report,
    save_markdown_report,
)


def build_mixed_state() -> dict:
    return {
        "dataset_path": (
            "data/student_performance.csv"
        ),
        "question": (
            "请解释 passed 字段含义，"
            "并统计不同 passed 值的平均 score。"
        ),
        "task_type": "mixed_analysis",
        "workflow_status": "completed",
        "step_count": 2,
        "plan": [
            "检索 passed 字段定义",
            "统计不同 passed 值的平均 score",
            "汇总工具结果",
        ],
        "tool_calls": [
            {
                "id": "call_rag",
                "tool_name": "retrieve_docs",
                "arguments": {
                    "query": (
                        "请解释 passed 字段含义"
                    ),
                    "top_k": 3,
                },
            },
            {
                "id": "call_sql",
                "tool_name": "run_readonly_sql",
                "arguments": {
                    "dataset_path": (
                        "data/student_performance.csv"
                    ),
                    "query": (
                        "SELECT passed, AVG(score) "
                        "AS avg_score FROM dataset "
                        "GROUP BY passed"
                    ),
                },
            },
        ],
        "tool_results": [
            {
                "tool_name": "retrieve_docs",
                "success": True,
                "result": {
                    "query": (
                        "请解释 passed 字段含义"
                    ),
                    "result_count": 3,
                },
                "error": None,
            },
            {
                "tool_name": "run_readonly_sql",
                "success": True,
                "result": {
                    "columns": [
                        "passed",
                        "avg_score",
                    ],
                    "rows": [
                        {
                            "passed": 0,
                            "avg_score": 62.09,
                        },
                        {
                            "passed": 1,
                            "avg_score": 84.76,
                        },
                    ],
                },
                "error": None,
            },
        ],
        "final_answer": (
            "passed=0 时平均 score 约为 62.09，"
            "passed=1 时约为 84.76。"
        ),
        "error": None,
    }


def test_generate_markdown_report_contains_core_sections() -> None:
    state = build_mixed_state()

    fixed_time = datetime(
        2026,
        7,
        28,
        16,
        0,
        0,
        tzinfo=timezone.utc,
    )

    report = generate_markdown_report(
        state,
        generated_at=fixed_time,
    )

    # 检查报告核心章节。
    assert (
        "# CausaLens Analysis Report"
        in report
    )
    assert "## Execution Summary" in report
    assert "## User Question" in report
    assert "## Analysis Plan" in report
    assert "## Evidence Summary" in report
    assert "## Tool Execution" in report
    assert "## Final Answer" in report

    # 检查执行状态。
    assert "`mixed_analysis`" in report
    assert "`completed`" in report
    assert "Step count: 2" in report

    # 检查面向用户的证据摘要。
    assert (
        "`retrieve_docs`：成功"
        in report
    )
    assert (
        "`run_readonly_sql`：成功"
        in report
    )
    assert (
        "检索问题为“请解释 passed 字段含义”"
        in report
    )
    assert "返回 3 个片段" in report
    assert "返回 2 行" in report
    assert (
        "结果字段为 passed、avg_score"
        in report
    )

    # 检查保留的原始工具调用与结果。
    assert "### 1. `retrieve_docs`" in report
    assert "### 2. `run_readonly_sql`" in report
    assert '"top_k": 3' in report
    assert "AVG(score)" in report

    assert "62.09" in report
    assert "84.76" in report

def test_save_markdown_report_creates_file(
    tmp_path: Path,
) -> None:
    state = build_mixed_state()

    output_path = (
        tmp_path
        / "nested"
        / "analysis_report.md"
    )

    saved_path = save_markdown_report(
        state,
        output_path,
    )

    assert saved_path == output_path
    assert output_path.exists()

    content = output_path.read_text(
        encoding="utf-8"
    )

    assert (
        "# CausaLens Analysis Report"
        in content
    )
    assert "retrieve_docs" in content
    assert "run_readonly_sql" in content


def test_report_handles_unsupported_task() -> None:
    state = {
        "dataset_path": "data/sample.csv",
        "question": "请删除所有数据。",
        "task_type": "unsupported",
        "workflow_status": "completed",
        "step_count": 0,
        "plan": [],
        "tool_calls": [],
        "tool_results": [],
        "final_answer": (
            "当前项目只支持只读分析，"
            "不会删除原始数据。"
        ),
        "error": None,
    }

    report = generate_markdown_report(
        state
    )

    assert "`unsupported`" in report
    assert "No tools were called." in report
    assert "不会删除原始数据" in report


def test_report_chart_summary_uses_returned_facts_only() -> None:
    state = {
        "dataset_path": (
            "data/student_performance.csv"
        ),
        "question": (
            "请画出 study_hours "
            "和 score 的散点图。"
        ),
        "task_type": "chart_analysis",
        "workflow_status": "completed",
        "step_count": 1,
        "plan": [
            "生成散点图",
            "总结工具结果",
        ],
        "tool_calls": [
            {
                "id": "call_chart",
                "tool_name": "create_chart",
                "arguments": {
                    "dataset_path": (
                        "data/student_performance.csv"
                    ),
                    "chart_type": "scatter",
                    "x_column": "study_hours",
                    "y_column": "score",
                },
            }
        ],
        "tool_results": [
            {
                "tool_name": "create_chart",
                "success": True,
                "result": {
                    "chart_type": "scatter",
                    "plotted_row_count": 40,
                    "chart_path": (
                        "outputs/charts/"
                        "study_hours_score.png"
                    ),
                    "relationship_summary": {
                        "point_count": 40,
                        "x_min": 3.0,
                        "x_max": 18.0,
                        "y_min": 52.0,
                        "y_max": 98.0,
                        "pearson_correlation": (
                            0.9874
                        ),
                    },
                },
                "error": None,
            }
        ],
        "final_answer": (
            "散点图已经生成。"
        ),
        "error": None,
    }

    report = generate_markdown_report(
        state
    )

    assert "## Evidence Summary" in report
    assert "生成 scatter 图" in report
    assert (
        "参与绘图数据为 40 条"
        in report
    )
    assert (
        "pearson_correlation=0.9874"
        in report
    )
    assert (
        "study_hours_score.png"
        in report
    )

    # 报告摘要本身不根据相关系数追加解释。
    assert "强相关" not in report
    assert "弱相关" not in report
    assert "导致成绩提高" not in report


def test_report_summarizes_failed_tool() -> None:
    state = {
        "dataset_path": "data/sample.csv",
        "question": "统计不存在的字段。",
        "task_type": "sql_analysis",
        "workflow_status": "failed",
        "step_count": 1,
        "plan": [
            "执行只读 SQL",
        ],
        "tool_calls": [
            {
                "id": "call_sql",
                "tool_name": (
                    "run_readonly_sql"
                ),
                "arguments": {
                    "dataset_path": (
                        "data/sample.csv"
                    ),
                    "query": (
                        "SELECT unknown_column "
                        "FROM dataset"
                    ),
                },
            }
        ],
        "tool_results": [
            {
                "tool_name": (
                    "run_readonly_sql"
                ),
                "success": False,
                "result": None,
                "error": (
                    "字段 unknown_column 不存在。"
                ),
            }
        ],
        "final_answer": (
            "无法完成当前查询。"
        ),
        "error": (
            "字段 unknown_column 不存在。"
        ),
    }

    report = generate_markdown_report(
        state
    )

    assert (
        "`run_readonly_sql`：失败"
        in report
    )
    assert (
        "字段 unknown_column 不存在"
        in report
    )
    assert "## Workflow Error" in report