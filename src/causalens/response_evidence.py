from __future__ import annotations

from typing import Any

from causalens.schemas import ToolResult


def _format_number(value: Any) -> str:
    """
    将数值转为适合 CLI 展示的简短文本。
    """
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")

    return str(value)


def _format_columns(
    column_names: list[Any],
    max_columns: int = 6,
) -> str:
    """
    只展示前几个字段名，避免 CLI 表格过长。
    """
    names = [str(column_name) for column_name in column_names]

    if not names:
        return "未返回字段信息"

    visible_names = names[:max_columns]
    text = "、".join(visible_names)

    if len(names) > max_columns:
        text += "……"

    return text


def _build_profile_evidence(
    result: dict[str, Any],
) -> str:
    rows = result.get("rows", "未知")
    columns = result.get("columns", "未知")

    column_names = result.get("column_names", [])
    columns_text = _format_columns(column_names)

    return (
        f"数据集规模：{rows} 行 × {columns} 列；"
        f"字段：{columns_text}"
    )


def _build_sql_evidence(
    result: dict[str, Any],
) -> str:
    returned_row_count = result.get(
        "returned_row_count",
        result.get("row_count", "未知"),
    )

    columns = result.get("columns", [])
    columns_text = _format_columns(columns)

    truncated = result.get("truncated", False)
    truncation_text = "；结果已截断" if truncated else ""

    return (
        f"SQL 返回 {returned_row_count} 行；"
        f"结果字段：{columns_text}"
        f"{truncation_text}"
    )


def _build_chart_evidence(
    result: dict[str, Any],
) -> str:
    chart_type = result.get("chart_type", "未知")
    plotted_row_count = result.get("plotted_row_count")

    parts = [f"生成 {chart_type} 图"]

    if plotted_row_count is not None:
        parts.append(f"参与绘图数据：{plotted_row_count} 条")

    numeric_summary = result.get("numeric_summary")

    if isinstance(numeric_summary, dict):
        numeric_parts = []

        for key in ["count", "mean", "median", "min", "max"]:
            if key in numeric_summary:
                numeric_parts.append(
                    f"{key}={_format_number(numeric_summary[key])}"
                )

        if numeric_parts:
            parts.append(
                "统计摘要：" + "，".join(numeric_parts)
            )

    relationship_summary = result.get("relationship_summary")

    if isinstance(relationship_summary, dict):
        relationship_parts = []

        for key in [
            "point_count",
            "x_min",
            "x_max",
            "y_min",
            "y_max",
            "pearson_correlation",
        ]:
            if key in relationship_summary:
                relationship_parts.append(
                    f"{key}="
                    f"{_format_number(relationship_summary[key])}"
                )

        if relationship_parts:
            parts.append(
                "关系摘要：" + "，".join(relationship_parts)
            )

    chart_path = result.get("chart_path")

    if chart_path:
        parts.append(f"图表：{chart_path}")

    return "\n".join(parts)


def build_evidence_rows(
    tool_results: list[ToolResult],
) -> list[dict[str, str]]:
    """
    将 Agent 的 ToolResult 转换为适合 CLI 展示的证据行。

    返回的每一项都只使用工具实际返回的结构化数据，
    不做额外统计推断或自然语言扩写。
    """
    evidence_rows: list[dict[str, str]] = []

    for tool_result in tool_results:
        if not tool_result.success:
            evidence_rows.append(
                {
                    "tool_name": tool_result.tool_name,
                    "status": "失败",
                    "evidence": (
                        f"错误：{tool_result.error or '未知错误'}"
                    ),
                }
            )
            continue

        result = tool_result.result or {}

        if tool_result.tool_name == "profile_dataset":
            evidence = _build_profile_evidence(result)

        elif tool_result.tool_name == "run_readonly_sql":
            evidence = _build_sql_evidence(result)

        elif tool_result.tool_name == "create_chart":
            evidence = _build_chart_evidence(result)

        else:
            result_keys = _format_columns(
                list(result.keys()),
            )

            evidence = f"返回字段：{result_keys}"

        evidence_rows.append(
            {
                "tool_name": tool_result.tool_name,
                "status": "成功",
                "evidence": evidence,
            }
        )

    return evidence_rows