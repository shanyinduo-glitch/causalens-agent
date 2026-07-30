from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any


def _to_plain_value(value: Any) -> Any:
    """
    将 Pydantic 模型、Path 和嵌套对象转换为可 JSON 序列化的普通值。
    """

    if hasattr(value, "model_dump"):
        return _to_plain_value(
            value.model_dump()
        )

    if isinstance(value, Mapping):
        return {
            str(key): _to_plain_value(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            _to_plain_value(item)
            for item in value
        ]

    if isinstance(value, Path):
        return str(value)

    if value is None or isinstance(
        value,
        (str, int, float, bool),
    ):
        return value

    return str(value)


def _format_json(value: Any) -> str:
    """
    将工具参数或结果格式化为 Markdown JSON 代码块。
    """

    plain_value = _to_plain_value(value)

    return json.dumps(
        plain_value,
        ensure_ascii=False,
        indent=2,
    )


def _normalize_tool_call(
    tool_call: Any,
) -> dict[str, Any]:
    """
    兼容字典和对象形式的工具调用记录。
    """

    plain_call = _to_plain_value(tool_call)

    if not isinstance(plain_call, dict):
        return {
            "id": None,
            "name": "unknown_tool",
            "arguments": {},
        }

    tool_name = (
        plain_call.get("tool_name")
        or plain_call.get("name")
        or "unknown_tool"
    )

    arguments = (
        plain_call.get("arguments")
        or plain_call.get("args")
        or {}
    )

    return {
        "id": plain_call.get("id"),
        "name": str(tool_name),
        "arguments": arguments,
    }


def _normalize_tool_result(
    tool_result: Any,
) -> dict[str, Any]:
    """
    兼容字典和 Pydantic ToolResult。
    """

    plain_result = _to_plain_value(
        tool_result
    )

    if not isinstance(plain_result, dict):
        return {
            "tool_name": "unknown_tool",
            "success": False,
            "result": None,
            "error": (
                "无法解析工具结果。"
            ),
        }

    tool_name = (
        plain_result.get("tool_name")
        or plain_result.get("name")
        or "unknown_tool"
    )

    return {
        "tool_name": str(tool_name),
        "success": (
            plain_result.get("success")
            is True
        ),
        "result": plain_result.get(
            "result"
        ),
        "error": plain_result.get(
            "error"
        ),
    }


def _match_tool_results(
    tool_calls: list[dict[str, Any]],
    tool_results: list[dict[str, Any]],
) -> list[
    tuple[
        dict[str, Any],
        dict[str, Any] | None,
    ]
]:
    """
    按工具名称为调用记录匹配对应结果。

    在名称无法匹配时，使用尚未匹配的下一条结果，
    避免报告因记录格式变化而崩溃。
    """

    matched_items: list[
        tuple[
            dict[str, Any],
            dict[str, Any] | None,
        ]
    ] = []

    used_result_indexes: set[int] = set()

    for tool_call in tool_calls:
        matched_index: int | None = None

        for index, tool_result in enumerate(
            tool_results
        ):
            if index in used_result_indexes:
                continue

            if (
                tool_result["tool_name"]
                == tool_call["name"]
            ):
                matched_index = index
                break

        if matched_index is None:
            for index in range(
                len(tool_results)
            ):
                if index not in used_result_indexes:
                    matched_index = index
                    break

        if matched_index is None:
            matched_result = None
        else:
            used_result_indexes.add(
                matched_index
            )
            matched_result = tool_results[
                matched_index
            ]

        matched_items.append(
            (
                tool_call,
                matched_result,
            )
        )

    return matched_items


def _format_compact_value(
    value: Any,
) -> str:
    """
    将数值格式化为适合摘要展示的紧凑文本。
    """

    if isinstance(value, float):
        return (
            f"{value:.4f}"
            .rstrip("0")
            .rstrip(".")
        )

    return str(value)


def _extract_document_sources(
    result: Mapping[str, Any],
) -> list[str]:
    """
    从 RAG 结果中提取来源文件和章节。

    最多保留前三个不同来源，避免摘要过长。
    """

    raw_results = result.get("results")

    if not isinstance(raw_results, list):
        return []

    source_labels: list[str] = []

    for item in raw_results:
        if not isinstance(item, Mapping):
            continue

        metadata = item.get("metadata")

        if not isinstance(metadata, Mapping):
            metadata = {}

        source = (
            item.get("source")
            or item.get("source_file")
            or item.get("document_name")
            or metadata.get("source")
            or metadata.get("source_file")
        )

        section = (
            item.get("section")
            or metadata.get("section")
        )

        if not source:
            continue

        label = str(source)

        if section:
            label = (
                f"{label} / {section}"
            )

        if label not in source_labels:
            source_labels.append(label)

        if len(source_labels) >= 3:
            break

    return source_labels


def _summarize_tool_result(
    tool_result: Mapping[str, Any],
) -> str:
    """
    将单条工具结果转换为确定性的 Markdown 摘要。

    只整理工具实际返回的字段，
    不补充业务解释或统计推断。
    """

    tool_name = str(
        tool_result.get(
            "tool_name",
            "unknown_tool",
        )
    )

    success = (
        tool_result.get("success")
        is True
    )

    if not success:
        error = (
            tool_result.get("error")
            or "未提供具体错误信息"
        )

        return (
            f"- `{tool_name}`：失败；"
            f"{error}"
        )

    result = tool_result.get("result")

    if not isinstance(result, Mapping):
        return (
            f"- `{tool_name}`：成功；"
            "工具未返回结构化结果。"
        )

    parts: list[str] = ["成功"]

    if tool_name == "profile_dataset":
        row_count = (
            result.get("row_count")
            or result.get("rows")
        )
        column_names = (
            result.get("column_names")
            or result.get("columns")
        )

        if isinstance(row_count, int):
            parts.append(
                f"数据共 {row_count} 行"
            )

        if isinstance(column_names, list):
            parts.append(
                f"共 {len(column_names)} 个字段"
            )
            parts.append(
                "字段为 "
                + "、".join(
                    str(name)
                    for name in column_names
                )
            )

    elif tool_name == "retrieve_docs":
        query = result.get("query")
        raw_results = result.get("results")
        result_count = result.get(
            "result_count"
        )

        if (
            result_count is None
            and isinstance(raw_results, list)
        ):
            result_count = len(raw_results)

        if query:
            parts.append(
                f"检索问题为“{query}”"
            )

        if isinstance(result_count, int):
            parts.append(
                f"返回 {result_count} 个片段"
            )

        source_labels = (
            _extract_document_sources(
                result
            )
        )

        if source_labels:
            parts.append(
                "来源包括 "
                + "、".join(source_labels)
            )

    elif tool_name == "run_readonly_sql":
        rows = result.get("rows")
        columns = result.get("columns")
        row_count = result.get(
            "row_count"
        )

        if (
            row_count is None
            and isinstance(rows, list)
        ):
            row_count = len(rows)

        if isinstance(row_count, int):
            parts.append(
                f"返回 {row_count} 行"
            )

        if isinstance(columns, list):
            parts.append(
                "结果字段为 "
                + "、".join(
                    str(column)
                    for column in columns
                )
            )

    elif tool_name == "create_chart":
        chart_type = result.get(
            "chart_type"
        )
        plotted_row_count = result.get(
            "plotted_row_count"
        )
        chart_path = result.get(
            "chart_path"
        )

        if chart_type:
            parts.append(
                f"生成 {chart_type} 图"
            )

        if isinstance(
            plotted_row_count,
            int,
        ):
            parts.append(
                f"参与绘图数据为 "
                f"{plotted_row_count} 条"
            )

        numeric_summary = result.get(
            "numeric_summary"
        )

        if isinstance(
            numeric_summary,
            Mapping,
        ):
            summary_parts = []

            for key in (
                "count",
                "mean",
                "median",
                "min",
                "max",
            ):
                if key in numeric_summary:
                    summary_parts.append(
                        f"{key}="
                        f"{_format_compact_value(numeric_summary[key])}"
                    )

            if summary_parts:
                parts.append(
                    "数值摘要为 "
                    + "，".join(summary_parts)
                )

        relationship_summary = (
            result.get(
                "relationship_summary"
            )
        )

        if isinstance(
            relationship_summary,
            Mapping,
        ):
            relationship_parts = []

            for key in (
                "point_count",
                "x_min",
                "x_max",
                "y_min",
                "y_max",
                "pearson_correlation",
            ):
                if key in relationship_summary:
                    relationship_parts.append(
                        f"{key}="
                        f"{_format_compact_value(relationship_summary[key])}"
                    )

            if relationship_parts:
                parts.append(
                    "关系摘要为 "
                    + "，".join(
                        relationship_parts
                    )
                )

        if chart_path:
            parts.append(
                f"图表路径为 `{chart_path}`"
            )

    return (
        f"- `{tool_name}`："
        + "；".join(parts)
        + "。"
    )


def generate_markdown_report(
    state: Mapping[str, Any],
    *,
    dataset_path: str | Path | None = None,
    generated_at: datetime | None = None,
) -> str:
    """
    根据最终 AgentState 生成 Markdown 分析报告。

    本函数只负责整理和保存现有证据，
    不重新解释、补充或推断工具结果。
    """

    report_time = (
        generated_at
        or datetime.now().astimezone()
    )

    question = str(
        state.get("question") or "未提供"
    )
    task_type = str(
        state.get("task_type") or "unknown"
    )
    workflow_status = str(
        state.get("workflow_status")
        or "unknown"
    )
    final_answer = str(
        state.get("final_answer")
        or "未生成最终回答。"
    )

    resolved_dataset_path = (
        dataset_path
        or state.get("dataset_path")
        or "未提供"
    )

    plan = state.get("plan") or []

    if not isinstance(plan, list):
        plan = [str(plan)]

    raw_tool_calls = (
        state.get("tool_calls") or []
    )
    raw_tool_results = (
        state.get("tool_results") or []
    )

    if not isinstance(
        raw_tool_calls,
        list,
    ):
        raw_tool_calls = []

    if not isinstance(
        raw_tool_results,
        list,
    ):
        raw_tool_results = []

    tool_calls = [
        _normalize_tool_call(tool_call)
        for tool_call in raw_tool_calls
    ]

    tool_results = [
        _normalize_tool_result(tool_result)
        for tool_result in raw_tool_results
    ]

    matched_tools = _match_tool_results(
        tool_calls,
        tool_results,
    )

    successful_tool_count = sum(
        tool_result["success"]
        for tool_result in tool_results
    )

    failed_tool_count = sum(
        not tool_result["success"]
        for tool_result in tool_results
    )

    lines = [
        "# CausaLens Analysis Report",
        "",
        "## Execution Summary",
        "",
        (
            f"- Generated at: "
            f"{report_time.isoformat(timespec='seconds')}"
        ),
        (
            f"- Dataset: "
            f"`{resolved_dataset_path}`"
        ),
        f"- Task type: `{task_type}`",
        (
            f"- Workflow status: "
            f"`{workflow_status}`"
        ),
        (
            f"- Step count: "
            f"{state.get('step_count', 0)}"
        ),
        (
            f"- Tool calls: "
            f"{len(tool_calls)}"
        ),
        (
            f"- Successful tools: "
            f"{successful_tool_count}"
        ),
        (
            f"- Failed tools: "
            f"{failed_tool_count}"
        ),
        "",
        "## User Question",
        "",
        question,
        "",
        "## Analysis Plan",
        "",
    ]

    if plan:
        for index, step in enumerate(
            plan,
            start=1,
        ):
            lines.append(
                f"{index}. {step}"
            )
    else:
        lines.append(
            "No analysis plan was generated."
        )

    lines.extend(
        [
            "",
            "## Evidence Summary",
            "",
        ]
    )

    if tool_results:
        for tool_result in tool_results:
            lines.append(
                _summarize_tool_result(
                    tool_result
                )
            )
    else:
        lines.append(
            "No tool evidence was produced."
        )

    lines.extend(
        [
            "",
            "## Tool Execution",
            "",
        ]
    )

    if not matched_tools:
        lines.append(
            "No tools were called."
        )

    for index, (
        tool_call,
        tool_result,
    ) in enumerate(
        matched_tools,
        start=1,
    ):
        tool_name = tool_call["name"]

        lines.extend(
            [
                f"### {index}. `{tool_name}`",
                "",
            ]
        )

        if tool_call["id"]:
            lines.append(
                f"- Call ID: `{tool_call['id']}`"
            )

        if tool_result is None:
            lines.append(
                "- Status: Result not found"
            )
        elif tool_result["success"]:
            lines.append(
                "- Status: Success"
            )
        else:
            lines.append(
                "- Status: Failed"
            )

        lines.extend(
            [
                "",
                "**Arguments**",
                "",
                "```json",
                _format_json(
                    tool_call["arguments"]
                ),
                "```",
                "",
            ]
        )

        if tool_result is None:
            lines.extend(
                [
                    "**Result**",
                    "",
                    "No matching tool result was found.",
                    "",
                ]
            )
            continue

        if tool_result["result"] is not None:
            lines.extend(
                [
                    "**Result**",
                    "",
                    "```json",
                    _format_json(
                        tool_result["result"]
                    ),
                    "```",
                    "",
                ]
            )

        if tool_result["error"]:
            lines.extend(
                [
                    "**Error**",
                    "",
                    str(tool_result["error"]),
                    "",
                ]
            )

    lines.extend(
        [
            "## Final Answer",
            "",
            final_answer,
            "",
        ]
    )

    state_error = state.get("error")

    if state_error:
        lines.extend(
            [
                "## Workflow Error",
                "",
                str(state_error),
                "",
            ]
        )

    return "\n".join(lines).rstrip() + "\n"


def save_markdown_report(
    state: Mapping[str, Any],
    output_path: str | Path,
    *,
    dataset_path: str | Path | None = None,
    generated_at: datetime | None = None,
) -> Path:
    """
    生成报告并写入指定 Markdown 文件。
    """

    resolved_output_path = Path(
        output_path
    )

    resolved_output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report = generate_markdown_report(
        state,
        dataset_path=dataset_path,
        generated_at=generated_at,
    )

    resolved_output_path.write_text(
        report,
        encoding="utf-8",
    )

    return resolved_output_path