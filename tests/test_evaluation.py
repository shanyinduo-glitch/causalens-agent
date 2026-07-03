import json
from pathlib import Path

from causalens.evaluation import (
    evaluate_agent_response,
    load_evaluation_cases,
)
from causalens.schemas import (
    AgentResponse,
    ToolCall,
    ToolResult,
)


def test_load_evaluation_cases_reads_jsonl(
    tmp_path: Path,
):
    case_path = tmp_path / "cases.jsonl"

    case_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": "case_01",
                        "question": "这个数据集有哪些字段？",
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "id": "case_02",
                        "question": "不同专业平均分是多少？",
                    },
                    ensure_ascii=False,
                ),
            ]
        ),
        encoding="utf-8",
    )

    cases = load_evaluation_cases(case_path)

    assert len(cases) == 2
    assert cases[0]["id"] == "case_01"
    assert cases[1]["question"] == (
        "不同专业平均分是多少？"
    )


def test_evaluate_agent_response_passes_chart_case():
    case = {
        "id": "chart_case",
        "category": "chart",
        "question": "请画出不同专业平均 score 的柱状图。",
        "required_tools": ["create_chart"],
        "forbidden_tools": [],
        "expected_chart_type": "bar",
    }

    response = AgentResponse(
        answer="已生成柱状图。",
        tool_calls=[
            ToolCall(
                tool_name="create_chart",
                arguments={
                    "dataset_path": (
                        "data/student_performance.csv"
                    ),
                    "chart_type": "bar",
                    "x_column": "major",
                    "y_column": "score",
                    "aggregation": "mean",
                },
            )
        ],
        tool_results=[
            ToolResult(
                tool_name="create_chart",
                success=True,
                result={
                    "chart_type": "bar",
                },
            )
        ],
    )

    evaluation = evaluate_agent_response(
        case,
        response,
    )

    assert evaluation["passed"] is True
    assert evaluation["checks"] == {
        "required_tools_called": True,
        "required_tools_succeeded": True,
        "expected_chart_type": True,
    }

    assert evaluation["actual_tools"] == [
        "create_chart",
    ]


def test_evaluate_agent_response_detects_forbidden_tool():
    case = {
        "id": "boundary_case",
        "category": "boundary",
        "question": "请画出 major 和 city 的散点图。",
        "required_tools": [],
        "forbidden_tools": ["create_chart"],
    }

    response = AgentResponse(
        answer="已生成散点图。",
        tool_calls=[
            ToolCall(
                tool_name="create_chart",
                arguments={
                    "chart_type": "scatter",
                    "x_column": "major",
                    "y_column": "city",
                },
            )
        ],
        tool_results=[
            ToolResult(
                tool_name="create_chart",
                success=False,
                error="x_column=major 必须是数值列。",
            )
        ],
    )

    evaluation = evaluate_agent_response(
        case,
        response,
    )

    assert evaluation["passed"] is False
    assert evaluation["checks"] == {
        "forbidden_tools_not_called": False,
    }