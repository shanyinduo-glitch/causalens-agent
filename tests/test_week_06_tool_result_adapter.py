from causalens.agent.nodes import (
    _normalize_tool_result,
)
from causalens.schemas import ToolResult
from causalens.tools.statistical_tool import (
    run_correlation,
)


DATASET_PATH = (
    "data/student_performance.csv"
)


def test_preserves_existing_tool_result():
    original = ToolResult(
        tool_name="example_tool",
        success=True,
        result={
            "value": 1,
        },
    )

    normalized = _normalize_tool_result(
        tool_name="example_tool",
        raw_result=original,
    )

    assert normalized is original


def test_adapts_statistical_result():
    raw_result = run_correlation(
        dataset_path=DATASET_PATH,
        x_column="study_hours",
        y_column="score",
        method="pearson",
    )

    normalized = _normalize_tool_result(
        tool_name="run_correlation",
        raw_result=raw_result,
    )

    assert isinstance(
        normalized,
        ToolResult,
    )
    assert normalized.success is True
    assert normalized.error is None
    assert normalized.result is not None
    assert (
        normalized.result["analysis_type"]
        == "correlation"
    )
    assert (
        normalized.result["variables"]
        == [
            "study_hours",
            "score",
        ]
    )
    assert (
            "Correlation does not establish causality."
            in normalized.warnings
    )


def test_rejects_unsupported_return_type():
    normalized = _normalize_tool_result(
        tool_name="broken_tool",
        raw_result="unexpected text",
    )

    assert normalized.success is False
    assert normalized.error is not None
    assert "不支持的类型" in normalized.error