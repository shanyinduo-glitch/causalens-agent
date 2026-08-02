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


class ChartRequest(BaseModel):
    """
    图表工具的标准输入结构。

    这个模型的职责不是画图，
    而是先检查模型传来的图表参数是否符合项目规则。
    """

    dataset_path: str

    chart_type: Literal[
        "bar",
        "line",
        "scatter",
        "histogram",
    ]

    x_column: str
    y_column: str | None = None

    # 第一版先保留这个字段。
    # 后续更复杂的分组图、分面图可以使用它。
    group_by: str | None = None

    aggregation: Literal[
        "mean",
        "sum",
        "count",
    ] | None = None

    title: str | None = None

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

class StatisticalResult(BaseModel):
    """统计分析工具的统一结果结构。"""

    analysis_type: str
    variables: list[str]
    sample_size: int

    estimate: float | None = None
    statistic: float | None = None
    p_value: float | None = None
    confidence_interval: list[float] | None = None

    details: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)

class RegressionCoefficient(BaseModel):
    """单个回归项的估计结果。"""

    term: str
    estimate: float

    standard_error: float
    robust_standard_error: float

    p_value: float
    confidence_interval: list[float]


class RegressionResult(BaseModel):
    """OLS 回归工具的统一结果结构。"""

    analysis_type: str = "ols"

    outcome: str
    predictors: list[str]
    categorical_columns: list[str]

    sample_size: int
    dropped_rows: int

    covariance_type: str

    coefficients: list[RegressionCoefficient]

    r_squared: float
    adjusted_r_squared: float

    diagnostics: dict[str, Any] = Field(
        default_factory=dict
    )
    warnings: list[str] = Field(
        default_factory=list
    )

class CausalReadinessResult(BaseModel):
    """因果分析前提检查的结构化结果。"""

    analysis_type: str = "causal_readiness"

    ready: bool

    treatment: str
    outcome: str
    confounders: list[str]

    blocking_issues: list[str] = Field(
        default_factory=list
    )
    warnings: list[str] = Field(
        default_factory=list
    )

    sample_summary: dict[str, Any] = Field(
        default_factory=dict
    )
    variable_summary: dict[str, Any] = Field(
        default_factory=dict
    )
    thresholds: dict[str, Any] = Field(
        default_factory=dict
    )