from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError


class AnalysisRequest(BaseModel):
    """用户发给数据分析 Agent 的请求。"""

    dataset_path: str
    question: str


class ToolCall(BaseModel):
    """Agent 计划调用某个工具时的描述。"""

    tool_name: str
    arguments: dict[str, Any]


class ToolResult(BaseModel):
    """工具执行后的统一返回格式。"""

    tool_name: str
    success: bool
    result: dict[str, Any] | None = None
    error: str | None = None


class DatasetProfile(BaseModel):
    """数据集的基础画像。"""

    rows: int
    columns: int
    column_names: list[str]
    dtypes: dict[str, str]
    missing_values: dict[str, int]
    numeric_summary: dict[str, dict[str, float]]


class AgentResponse(BaseModel):
    """Agent 最终交给用户的结果。"""

    answer: str
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_results: list[ToolResult] = Field(default_factory=list)
    confidence: Literal["low", "medium", "high"] = "medium"


if __name__ == "__main__":
    print("=== 1. 正确请求 ===")

    request = AnalysisRequest(
        dataset_path="data/sample.csv",
        question="这个数据集有多少行？",
    )

    print(request.model_dump())

    print("\n=== 2. 错误请求 ===")

    try:
        bad_request = AnalysisRequest(
            dataset_path=123,
            question=None,
        )
        print(bad_request.model_dump())

    except ValidationError as error:
        print(type(error).__name__)
        print(error)

    print("\n=== 3. 工具调用计划 ===")

    tool_call = ToolCall(
        tool_name="profile_dataset",
        arguments={"dataset_path": "data/sample.csv"},
    )

    print(tool_call.model_dump())

    print("\n=== 4. 工具成功结果 ===")

    success_result = ToolResult(
        tool_name="profile_dataset",
        success=True,
        result={
            "rows": 6,
            "columns": 5,
        },
    )

    print(success_result.model_dump())

    print("\n=== 5. 工具失败结果 ===")

    failed_result = ToolResult(
        tool_name="profile_dataset",
        success=False,
        error="找不到文件：data/missing.csv",
    )

    print(failed_result.model_dump())

    print("\n=== 6. Agent 最终回答 ===")

    response = AgentResponse(
        answer="这个数据集共有 6 行、5 列。",
        tool_calls=[tool_call],
        tool_results=[success_result],
        confidence="high",
    )

    print(response.model_dump())