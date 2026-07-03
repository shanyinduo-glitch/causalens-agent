from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from causalens.schemas import AgentResponse


def load_evaluation_cases(
    case_path: Path,
) -> list[dict[str, Any]]:
    """
    从 JSONL 文件读取评测题。

    JSONL 的每一行都代表一个独立评测案例。
    """
    cases: list[dict[str, Any]] = []

    with case_path.open(
        "r",
        encoding="utf-8",
    ) as case_file:
        for line_number, raw_line in enumerate(
            case_file,
            start=1,
        ):
            line = raw_line.strip()

            if not line:
                continue

            case = json.loads(line)

            if "id" not in case:
                raise ValueError(
                    f"评测文件第 {line_number} 行缺少 id。"
                )

            if "question" not in case:
                raise ValueError(
                    f"评测文件第 {line_number} 行缺少 question。"
                )

            cases.append(case)

    return cases


def _build_tool_success_map(
    response: AgentResponse,
) -> dict[str, bool]:
    """
    记录每种工具是否至少成功执行过一次。
    """
    success_map: dict[str, bool] = {}

    for tool_result in response.tool_results:
        previous_success = success_map.get(
            tool_result.tool_name,
            False,
        )

        success_map[tool_result.tool_name] = (
            previous_success or tool_result.success
        )

    return success_map


def evaluate_agent_response(
    case: dict[str, Any],
    response: AgentResponse,
) -> dict[str, Any]:
    """
    根据评测案例要求，检查 Agent 的工具选择与执行结果。

    这里不尝试自动判断自然语言答案“写得好不好”。
    它只评估可稳定验证的事实：
    - 是否调用了应调用的工具；
    - 是否调用了不应调用的工具；
    - 目标工具是否成功；
    - 图表类型是否正确。
    """
    actual_tool_calls = [
        tool_call.tool_name
        for tool_call in response.tool_calls
    ]

    tool_success_map = _build_tool_success_map(response)

    required_tools = case.get(
        "required_tools",
        [],
    )

    forbidden_tools = case.get(
        "forbidden_tools",
        [],
    )

    checks: dict[str, bool] = {}

    if required_tools:
        checks["required_tools_called"] = all(
            tool_name in actual_tool_calls
            for tool_name in required_tools
        )

        checks["required_tools_succeeded"] = all(
            tool_success_map.get(tool_name, False)
            for tool_name in required_tools
        )

    if forbidden_tools:
        checks["forbidden_tools_not_called"] = all(
            tool_name not in actual_tool_calls
            for tool_name in forbidden_tools
        )

    if case.get("expect_no_tools", False):
        checks["no_tools_called"] = not actual_tool_calls

    expected_chart_type = case.get(
        "expected_chart_type",
    )

    if expected_chart_type is not None:
        chart_type_matches = any(
            tool_call.tool_name == "create_chart"
            and tool_call.arguments.get("chart_type")
            == expected_chart_type
            for tool_call in response.tool_calls
        )

        checks["expected_chart_type"] = chart_type_matches

    passed = all(checks.values()) if checks else True

    return {
        "id": case["id"],
        "category": case.get("category", "unknown"),
        "question": case["question"],
        "passed": passed,
        "checks": checks,
        "expected_tools": required_tools,
        "forbidden_tools": forbidden_tools,
        "expected_chart_type": expected_chart_type,
        "actual_tools": actual_tool_calls,
        "tool_success_map": tool_success_map,
        "answer": response.answer,
    }