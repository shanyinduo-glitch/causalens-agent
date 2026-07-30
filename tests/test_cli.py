from typer.testing import CliRunner

from causalens import cli
from pathlib import Path
from types import SimpleNamespace



runner = CliRunner()


def test_cli_shows_friendly_error_for_missing_file():
    """文件不存在时，CLI 应显示友好提示，而不是 Traceback。"""

    result = runner.invoke(
        cli.app,
        [
            "ask",
            "--dataset-path",
            "data/not_exists.csv",
            "--question",
            "这个数据有多少行？",
        ],
    )

    assert result.exit_code == 1
    assert "错误：找不到数据文件：data/not_exists.csv" in result.output
    assert "Traceback" not in result.output


def test_cli_prints_answer_when_agent_succeeds(
    monkeypatch,
    tmp_path,
):
    """
    LangGraph 正常返回最终状态时，
    CLI 应打印任务信息、最终回答和工具依据。
    """

    csv_path = tmp_path / "students.csv"
    csv_path.write_text(
        "student_id,score\n"
        "1,88\n"
        "2,92\n",
        encoding="utf-8",
    )

    def fake_run_causalens_graph(
        dataset_path: str,
        question: str,
    ) -> dict:
        return {
            "dataset_path": dataset_path,
            "question": question,
            "task_type": "data_profile",
            "workflow_status": "completed",
            "plan": [
                "读取 CSV 数据集概况",
                "总结数据集基本信息",
            ],
            "tool_calls": [
                {
                    "tool_name": "profile_dataset",
                    "arguments": {
                        "dataset_path": dataset_path,
                    },
                }
            ],
            "tool_results": [
                {
                    "tool_name": "profile_dataset",
                    "success": True,
                    "result": {
                        "rows": 2,
                        "columns": 2,
                        "column_names": [
                            "student_id",
                            "score",
                        ],
                    },
                    "error": None,
                }
            ],
            "final_answer": "这个数据集共有 2 行。",
            "error": None,
        }

    monkeypatch.setattr(
        cli,
        "run_causalens_graph",
        fake_run_causalens_graph,
    )

    result = runner.invoke(
        cli.app,
        [
            "ask",
            "--dataset-path",
            str(csv_path),
            "--question",
            "这个数据集有多少行？",
        ],
    )

    assert result.exit_code == 0
    assert "data_profile" in result.stdout
    assert "completed" in result.stdout
    assert "profile_dataset" in result.stdout
    assert "这个数据集共有 2 行。" in result.stdout
    assert "数据集规模：2 行 × 2 列" in result.stdout


def test_cli_ask_saves_markdown_report(
    monkeypatch,
    tmp_path: Path,
) -> None:
    fake_state = {
        "dataset_path": (
            "data/student_performance.csv"
        ),
        "question": (
            "不同专业的平均 score 分别是多少？"
        ),
        "task_type": "sql_analysis",
        "workflow_status": "completed",
        "step_count": 1,
        "plan": [
            "识别分组字段与数值字段",
            "执行只读 SQL",
            "汇总查询结果",
        ],
        "tool_calls": [
            {
                "id": "call_sql",
                "tool_name": "run_readonly_sql",
                "arguments": {
                    "dataset_path": (
                        "data/student_performance.csv"
                    ),
                    "query": (
                        "SELECT major, AVG(score) "
                        "AS avg_score "
                        "FROM dataset "
                        "GROUP BY major"
                    ),
                },
            }
        ],
        "tool_results": [
            {
                "tool_name": "run_readonly_sql",
                "success": True,
                "result": {
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
                "error": None,
            }
        ],
        "final_answer": (
            "cs 为 81.23，math 为 79.57，"
            "stats 为 74.69。"
        ),
        "error": None,
    }

    def fake_run_causalens_graph(
        dataset_path: str,
        question: str,
        max_steps: int = 5,
    ):
        return fake_state

    monkeypatch.setattr(
        cli,
        "run_causalens_graph",
        fake_run_causalens_graph,
    )

    report_path = (
        tmp_path
        / "reports"
        / "major_score.md"
    )

    runner = CliRunner()

    result = runner.invoke(
        cli.app,
        [
            "ask",
            "--dataset-path",
            "data/student_performance.csv",
            "--question",
            "不同专业的平均 score 分别是多少？",
            "--report-path",
            str(report_path),
        ],
    )

    assert result.exit_code == 0
    assert report_path.exists()
    assert "Markdown 报告已保存" in result.stdout

    report_content = report_path.read_text(
        encoding="utf-8"
    )

    assert (
        "# CausaLens Analysis Report"
        in report_content
    )
    assert "run_readonly_sql" in report_content
    assert "AVG(score)" in report_content
    assert "81.23" in report_content


def test_cli_ask_does_not_save_report_by_default(
    monkeypatch,
) -> None:
    fake_state = {
        "dataset_path": "data/sample.csv",
        "question": "这个数据集有哪些字段？",
        "task_type": "data_profile",
        "workflow_status": "completed",
        "step_count": 1,
        "plan": ["读取数据概览"],
        "tool_calls": [],
        "tool_results": [],
        "final_answer": "数据集包含多个字段。",
        "error": None,
    }

    def fake_run_causalens_graph(
        dataset_path: str,
        question: str,
        max_steps: int = 5,
    ):
        return fake_state

    monkeypatch.setattr(
        cli,
        "run_causalens_graph",
        fake_run_causalens_graph,
    )

    runner = CliRunner()

    result = runner.invoke(
        cli.app,
        [
            "ask",
            "--dataset-path",
            "data/sample.csv",
            "--question",
            "这个数据集有哪些字段？",
        ],
    )

    assert result.exit_code == 0
    assert (
        "Markdown 报告已保存"
        not in result.stdout
    )