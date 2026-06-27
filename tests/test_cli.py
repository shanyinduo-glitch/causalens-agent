from typer.testing import CliRunner

from causalens import cli
from causalens.schemas import AgentResponse, ToolCall, ToolResult


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


def test_cli_prints_answer_when_agent_succeeds(monkeypatch, tmp_path):
    """Agent 正常返回时，CLI 应打印最终答案和工具信息。"""

    csv_path = tmp_path / "students.csv"
    csv_path.write_text(
        "student_id,score\n"
        "1,88\n"
        "2,92\n",
        encoding="utf-8",
    )

    def fake_run_tool_agent(question: str, dataset_path: str) -> AgentResponse:
        return AgentResponse(
            answer="这个数据集共有 2 行。",
            tool_calls=[
                ToolCall(
                    tool_name="profile_dataset",
                    arguments={"dataset_path": dataset_path},
                )
            ],
            tool_results=[
                ToolResult(
                    tool_name="profile_dataset",
                    success=True,
                    result={"rows": 2, "columns": 2},
                )
            ],
            confidence="high",
        )

    monkeypatch.setattr(
        cli,
        "run_tool_agent",
        fake_run_tool_agent,
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
    assert "回答：" in result.output
    assert "这个数据集共有 2 行。" in result.output
    assert "profile_dataset" in result.output