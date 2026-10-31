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
    warnings: list[str] = Field(default_factory=list)


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


class CausalDataQualityResult(BaseModel):
    """因果估计前的只读数据质量门禁结果。"""

    analysis_type: str = "causal_data_quality"
    ready: bool
    required_columns: list[str]
    numeric_columns: list[str]
    blocking_issues: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    sample_summary: dict[str, Any] = Field(default_factory=dict)
    column_diagnostics: dict[str, Any] = Field(default_factory=dict)
    duplicate_diagnostics: dict[str, Any] = Field(default_factory=dict)
    preprocessing_policy: dict[str, Any] = Field(default_factory=dict)

class DMLResult(BaseModel):
    """LinearDML 因果探索的结构化结果。"""

    analysis_type: str = "linear_dml"

    treatment: str
    outcome: str
    confounders: list[str]

    sample_size: int
    discrete_treatment: bool

    ate: float
    confidence_interval: list[float]
    confidence_level: float = 0.95

    naive_group_difference: float | None = None

    model_configuration: dict[str, Any] = Field(
        default_factory=dict
    )
    readiness_summary: dict[str, Any] = Field(
        default_factory=dict
    )

    assumptions: list[str] = Field(
        default_factory=list
    )
    warnings: list[str] = Field(
        default_factory=list
    )


class AIPWResult(BaseModel):
    """交叉拟合 AIPW 平均处理效应估计结果。"""

    analysis_type: str = "aipw"
    estimator_name: str = "AIPW"

    treatment: str
    outcome: str
    confounders: list[str]
    sample_size: int
    discrete_treatment: bool = True

    ate: float
    standard_error: float
    confidence_interval: list[float]
    confidence_level: float = 0.95
    naive_group_difference: float | None = None

    model_configuration: dict[str, Any] = Field(default_factory=dict)
    readiness_summary: dict[str, Any] = Field(default_factory=dict)
    assumptions: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class CausalForestResult(BaseModel):
    """CausalForestDML 条件平均处理效应结果。"""

    analysis_type: str = "causal_forest_dml"
    estimator_name: str = "CausalForestDML"
    treatment: str
    outcome: str
    confounders: list[str]
    effect_modifiers: list[str]
    sample_size: int
    discrete_treatment: bool = True
    ate: float
    standard_error: float | None = None
    confidence_interval: list[float]
    confidence_level: float = 0.95
    naive_group_difference: float | None = None
    heterogeneity: dict[str, Any] = Field(default_factory=dict)
    model_configuration: dict[str, Any] = Field(default_factory=dict)
    readiness_summary: dict[str, Any] = Field(default_factory=dict)
    assumptions: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class DIDReadinessResult(BaseModel):
    """两期 Difference-in-Differences 的准备度结果。"""

    analysis_type: str = "did_readiness"
    ready: bool
    group_column: str
    post_column: str
    outcome: str
    confounders: list[str] = Field(default_factory=list)
    unit_id_column: str | None = None
    blocking_issues: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    sample_summary: dict[str, Any] = Field(default_factory=dict)
    thresholds: dict[str, Any] = Field(default_factory=dict)


class DIDResult(BaseModel):
    """两期面板或重复横截面 Difference-in-Differences 结果。"""

    analysis_type: str = "difference_in_differences"
    estimator_name: str = "DifferenceInDifferences"
    estimand: str = "ATT"
    group_column: str
    post_column: str
    outcome: str
    confounders: list[str] = Field(default_factory=list)
    unit_id_column: str | None = None
    sample_size: int
    ate: float
    standard_error: float
    confidence_interval: list[float]
    confidence_level: float = 0.95
    cell_means: dict[str, float] = Field(default_factory=dict)
    unadjusted_did: float
    model_configuration: dict[str, Any] = Field(default_factory=dict)
    readiness_summary: dict[str, Any] = Field(default_factory=dict)
    assumptions: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class EventStudyReadinessResult(BaseModel):
    """多期 Event Study 的准备度结果。"""

    analysis_type: str = "event_study_readiness"
    ready: bool
    group_column: str
    event_time_column: str
    outcome: str
    reference_period: int = -1
    unit_id_column: str | None = None
    blocking_issues: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    sample_summary: dict[str, Any] = Field(default_factory=dict)
    thresholds: dict[str, Any] = Field(default_factory=dict)


class EventStudyResult(BaseModel):
    """多期 Difference-in-Differences Event Study 结果。"""

    analysis_type: str = "event_study"
    estimator_name: str = "EventStudy"
    estimand: str = "ATT"
    group_column: str
    event_time_column: str
    outcome: str
    confounders: list[str] = Field(default_factory=list)
    unit_id_column: str | None = None
    reference_period: int = -1
    sample_size: int
    ate: float
    standard_error: float
    confidence_interval: list[float]
    confidence_level: float = 0.95
    dynamic_effects: list[dict[str, Any]] = Field(default_factory=list)
    pretrend_test: dict[str, Any] = Field(default_factory=dict)
    chart_path: str | None = None
    model_configuration: dict[str, Any] = Field(default_factory=dict)
    readiness_summary: dict[str, Any] = Field(default_factory=dict)
    assumptions: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class StaggeredDIDReadinessResult(BaseModel):
    """错位处理时点 cohort-time DiD 的准备度结果。"""

    analysis_type: str = "staggered_did_readiness"
    ready: bool
    treatment_column: str
    time_column: str
    treatment_start_column: str
    outcome: str
    unit_id_column: str
    blocking_issues: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    sample_summary: dict[str, Any] = Field(default_factory=dict)
    thresholds: dict[str, Any] = Field(default_factory=dict)


class StaggeredDIDResult(BaseModel):
    """使用尚未处理或永不处理对照的 cohort-time ATT 结果。"""

    analysis_type: str = "staggered_adoption_did"
    estimator_name: str = "StaggeredAdoptionDID"
    estimand: str = "ATT"
    treatment_column: str
    time_column: str
    treatment_start_column: str
    outcome: str
    unit_id_column: str
    confounders: list[str] = Field(default_factory=list)
    sample_size: int
    ate: float
    standard_error: float
    confidence_interval: list[float]
    confidence_level: float = 0.95
    group_time_effects: list[dict[str, Any]] = Field(default_factory=list)
    dynamic_effects: list[dict[str, Any]] = Field(default_factory=list)
    pretrend_test: dict[str, Any] = Field(default_factory=dict)
    placebo_test: dict[str, Any] = Field(default_factory=dict)
    cohort_sensitivity: dict[str, Any] = Field(default_factory=dict)
    chart_path: str | None = None
    model_configuration: dict[str, Any] = Field(default_factory=dict)
    readiness_summary: dict[str, Any] = Field(default_factory=dict)
    assumptions: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class IVReadinessResult(BaseModel):
    """工具变量设计的数据与第一阶段准备度。"""

    analysis_type: str = "iv_readiness"
    ready: bool
    treatment: str
    outcome: str
    instruments: list[str] = Field(default_factory=list)
    confounders: list[str] = Field(default_factory=list)
    sample_summary: dict[str, Any] = Field(default_factory=dict)
    first_stage: dict[str, Any] = Field(default_factory=dict)
    blocking_issues: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    thresholds: dict[str, Any] = Field(default_factory=dict)


class IVResult(BaseModel):
    """单一内生处理变量的异方差稳健 2SLS 结果。"""

    analysis_type: str = "instrumental_variable_2sls"
    estimator_name: str = "TwoStageLeastSquares"
    estimand: str = "LATE"
    treatment: str
    outcome: str
    instruments: list[str] = Field(default_factory=list)
    confounders: list[str] = Field(default_factory=list)
    sample_size: int
    ate: float
    standard_error: float
    confidence_interval: list[float]
    confidence_level: float = 0.95
    first_stage: dict[str, Any] = Field(default_factory=dict)
    endogeneity_test: dict[str, Any] = Field(default_factory=dict)
    overidentification_test: dict[str, Any] = Field(default_factory=dict)
    model_configuration: dict[str, Any] = Field(default_factory=dict)
    readiness_summary: dict[str, Any] = Field(default_factory=dict)
    assumptions: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class RDDReadinessResult(BaseModel):
    """Sharp RDD 的断点支持、分配一致性与伪检验准备度。"""

    analysis_type: str = "rdd_readiness"
    ready: bool
    treatment: str
    outcome: str
    running_variable: str
    cutoff: float
    treatment_above_cutoff: bool = True
    covariates: list[str] = Field(default_factory=list)
    bandwidth: float
    sample_summary: dict[str, Any] = Field(default_factory=dict)
    density_diagnostic: dict[str, Any] = Field(default_factory=dict)
    covariate_continuity: dict[str, Any] = Field(default_factory=dict)
    blocking_issues: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    thresholds: dict[str, Any] = Field(default_factory=dict)


class RDDResult(BaseModel):
    """带三角核与 HC1 推断的局部线性 sharp RDD。"""

    analysis_type: str = "sharp_regression_discontinuity"
    estimator_name: str = "SharpRegressionDiscontinuity"
    estimand: str = "LATE"
    treatment: str
    outcome: str
    running_variable: str
    cutoff: float
    treatment_above_cutoff: bool = True
    covariates: list[str] = Field(default_factory=list)
    bandwidth: float
    sample_size: int
    ate: float
    standard_error: float
    confidence_interval: list[float]
    confidence_level: float = 0.95
    p_value: float
    local_support: dict[str, Any] = Field(default_factory=dict)
    density_diagnostic: dict[str, Any] = Field(default_factory=dict)
    covariate_continuity: dict[str, Any] = Field(default_factory=dict)
    bandwidth_sensitivity: dict[str, Any] = Field(default_factory=dict)
    donut_sensitivity: dict[str, Any] = Field(default_factory=dict)
    model_configuration: dict[str, Any] = Field(default_factory=dict)
    readiness_summary: dict[str, Any] = Field(default_factory=dict)
    assumptions: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class SyntheticControlReadinessResult(BaseModel):
    """单处理单位合成控制的数据支持与处理时点检查。"""

    analysis_type: str = "synthetic_control_readiness"
    ready: bool
    treatment: str
    outcome: str
    unit_id_column: str
    time_column: str
    intervention_time: float
    treated_unit: str
    covariates: list[str] = Field(default_factory=list)
    blocking_issues: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    sample_summary: dict[str, Any] = Field(default_factory=dict)
    thresholds: dict[str, Any] = Field(default_factory=dict)


class SyntheticControlResult(BaseModel):
    """凸组合权重、处理后 ATT 与空间安慰剂推断结果。"""

    analysis_type: str = "synthetic_control"
    estimator_name: str = "SyntheticControl"
    estimand: str = "ATT"
    treatment: str
    outcome: str
    unit_id_column: str
    time_column: str
    intervention_time: float
    treated_unit: str
    covariates: list[str] = Field(default_factory=list)
    sample_size: int
    ate: float
    standard_error: float
    confidence_interval: list[float]
    confidence_level: float = 0.95
    p_value: float
    donor_weights: list[dict[str, Any]] = Field(default_factory=list)
    trajectories: list[dict[str, Any]] = Field(default_factory=list)
    fit_diagnostics: dict[str, Any] = Field(default_factory=dict)
    placebo_inference: dict[str, Any] = Field(default_factory=dict)
    model_configuration: dict[str, Any] = Field(default_factory=dict)
    readiness_summary: dict[str, Any] = Field(default_factory=dict)
    assumptions: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class TimeSeriesReadinessResult(BaseModel):
    """单序列连续冲击分析的时间顺序、样本量和稳定性检查。"""

    analysis_type: str = "time_series_readiness"
    ready: bool
    treatment: str
    outcome: str
    time_column: str
    confounders: list[str] = Field(default_factory=list)
    sample_summary: dict[str, Any] = Field(default_factory=dict)
    stationarity: dict[str, Any] = Field(default_factory=dict)
    blocking_issues: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    thresholds: dict[str, Any] = Field(default_factory=dict)


class TimeSeriesLocalProjectionResult(BaseModel):
    """带滞后控制和 HAC 推断的单序列局部投影动态响应。"""

    analysis_type: str = "time_series_local_projection"
    estimator_name: str = "TimeSeriesLocalProjection"
    estimand: str = "IRF"
    treatment: str
    outcome: str
    time_column: str
    confounders: list[str] = Field(default_factory=list)
    sample_size: int
    ate: float
    standard_error: float
    confidence_interval: list[float]
    confidence_level: float = 0.95
    p_value: float
    dynamic_effects: list[dict[str, Any]] = Field(default_factory=list)
    stationarity: dict[str, Any] = Field(default_factory=dict)
    placebo_test: dict[str, Any] = Field(default_factory=dict)
    residual_diagnostics: dict[str, Any] = Field(default_factory=dict)
    model_configuration: dict[str, Any] = Field(default_factory=dict)
    readiness_summary: dict[str, Any] = Field(default_factory=dict)
    assumptions: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class CausalEstimationSuiteResult(BaseModel):
    """主估计器与稳健性估计器的统一输出。"""

    analysis_type: str = "causal_estimation_suite"
    estimand: str = "ATE"
    design_type: str = "cross_sectional"
    primary_estimator: str
    robustness_estimators: list[str] = Field(default_factory=list)
    treatment: str
    outcome: str
    confounders: list[str]
    instruments: list[str] = Field(default_factory=list)
    running_variable: str | None = None
    cutoff: float | None = None
    treatment_above_cutoff: bool = True
    bandwidth: float | None = None
    intervention_time: float | None = None
    treated_unit: str | None = None
    sample_size: int
    estimates: list[dict[str, Any]] = Field(default_factory=list)
    comparison: dict[str, Any] = Field(default_factory=dict)
    sensitivity_summary: dict[str, Any] = Field(default_factory=dict)
    negative_control_summary: dict[str, Any] = Field(default_factory=dict)
    permutation_placebo: dict[str, Any] = Field(default_factory=dict)
    evidence_grade: dict[str, Any] = Field(default_factory=dict)
    data_quality: dict[str, Any] = Field(default_factory=dict)
    design_selection: dict[str, Any] = Field(default_factory=dict)
    runtime: dict[str, Any] = Field(default_factory=dict)
    assumptions: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
