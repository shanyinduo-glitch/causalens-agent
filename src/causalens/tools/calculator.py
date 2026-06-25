from causalens.schemas import ToolResult


def calculator(expression: str) -> ToolResult:
    """计算只包含基础四则运算的数学表达式。"""

    allowed_chars = set("0123456789+-*/(). ")

    if not set(expression) <= allowed_chars:
        return ToolResult(
            tool_name="calculator",
            success=False,
            error="Expression contains unsupported characters.",
        )

    try:
        value = eval(expression, {"__builtins__": {}}, {})

        return ToolResult(
            tool_name="calculator",
            success=True,
            result={
                "expression": expression,
                "value": value,
            },
        )

    except Exception as exc:
        return ToolResult(
            tool_name="calculator",
            success=False,
            error=str(exc),
        )