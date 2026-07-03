from pathlib import Path

import pytest

import causalens.observability as observability


@pytest.fixture(autouse=True)
def redirect_tool_call_logs_for_tests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """
    所有测试运行时，将工具日志重定向到 pytest 的临时目录。

    这样测试不会污染项目真实的 logs/tool_calls.jsonl。
    """
    monkeypatch.setattr(
        observability,
        "TOOL_CALL_LOG_PATH",
        tmp_path / "tool_calls.jsonl",
    )