from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


Estimand = Literal["ATE", "ATT", "ATC", "CATE", "LATE", "IRF"]
DesignStatus = Literal["draft", "awaiting_confirmation", "confirmed"]
StudyDesignType = Literal[
    "cross_sectional",
    "difference_in_differences",
    "instrumental_variable",
    "regression_discontinuity",
    "synthetic_control",
    "time_series",
]


class TreatmentContrast(BaseModel):
    """处理效应比较的两个处理水平。"""

    control: str | int | float | bool
    treated: str | int | float | bool

    @model_validator(mode="after")
    def validate_distinct_levels(self) -> "TreatmentContrast":
        if self.control == self.treated:
            raise ValueError(
                "control 和 treated 必须是不同的处理水平。"
            )
        return self


class VariableRoles(BaseModel):
    """研究设计中各变量的因果角色。"""

    confounders: list[str] = Field(default_factory=list)
    effect_modifiers: list[str] = Field(default_factory=list)
    mediators: list[str] = Field(default_factory=list)
    colliders: list[str] = Field(default_factory=list)
    instruments: list[str] = Field(default_factory=list)
    negative_control_outcomes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_roles(self) -> "VariableRoles":
        role_values = {
            "confounders": self.confounders,
            "effect_modifiers": self.effect_modifiers,
            "mediators": self.mediators,
            "colliders": self.colliders,
            "instruments": self.instruments,
            "negative_control_outcomes": self.negative_control_outcomes,
        }

        for role, values in role_values.items():
            normalized = [value.strip() for value in values]
            if any(not value for value in normalized):
                raise ValueError(f"{role} 不能包含空字段名。")
            if len(normalized) != len(set(normalized)):
                raise ValueError(f"{role} 不能包含重复字段。")
            setattr(self, role, normalized)

        forbidden_adjustment_roles = {
            "mediator": set(self.mediators),
            "collider": set(self.colliders),
            "instrument": set(self.instruments),
            "negative_control_outcome": set(self.negative_control_outcomes),
        }
        confounders = set(self.confounders)
        for role, variables in forbidden_adjustment_roles.items():
            overlap = sorted(confounders & variables)
            if overlap:
                raise ValueError(
                    "变量不能同时作为 confounder 和 "
                    f"{role}: {overlap}"
                )

        return self


class CausalQuestion(BaseModel):
    """经过用户确认后才能进入识别和估计阶段的因果问题契约。"""

    treatment: str
    outcome: str
    estimand: Estimand = "ATE"
    design_type: StudyDesignType = "cross_sectional"
    post_column: str | None = None
    event_time_column: str | None = None
    reference_period: int = -1
    time_column: str | None = None
    treatment_start_column: str | None = None
    unit_id_column: str | None = None
    running_variable: str | None = None
    cutoff: float | None = None
    treatment_above_cutoff: bool = True
    bandwidth: float | None = None
    intervention_time: float | None = None
    treated_unit: str | int | float | None = None
    max_horizon: int = 5
    treatment_lags: int = 1
    outcome_lags: int = 1
    control_lag: int = 1
    cumulative_outcome: bool = True
    hac_lags: int | None = None
    treatment_contrast: TreatmentContrast
    target_population: str
    unit_of_analysis: str
    outcome_horizon: str
    variable_roles: VariableRoles = Field(default_factory=VariableRoles)

    @model_validator(mode="after")
    def validate_question(self) -> "CausalQuestion":
        self.treatment = self.treatment.strip()
        self.outcome = self.outcome.strip()
        if not self.treatment or not self.outcome:
            raise ValueError("treatment 和 outcome 不能为空。")
        if self.treatment == self.outcome:
            raise ValueError(
                "treatment 和 outcome 不能是同一字段。"
            )

        reserved = {self.treatment, self.outcome}
        if self.post_column:
            self.post_column = self.post_column.strip()
            reserved.add(self.post_column)
        if self.event_time_column:
            self.event_time_column = self.event_time_column.strip()
            reserved.add(self.event_time_column)
        if self.time_column:
            self.time_column = self.time_column.strip()
            reserved.add(self.time_column)
        if self.treatment_start_column:
            self.treatment_start_column = self.treatment_start_column.strip()
            reserved.add(self.treatment_start_column)
        if self.unit_id_column:
            self.unit_id_column = self.unit_id_column.strip()
            reserved.add(self.unit_id_column)
        if self.running_variable:
            self.running_variable = self.running_variable.strip()
            reserved.add(self.running_variable)
        if self.design_type == "difference_in_differences":
            has_staggered_contract = bool(
                self.time_column and self.treatment_start_column and self.unit_id_column
            )
            if not self.post_column and not self.event_time_column and not has_staggered_contract:
                raise ValueError(
                    "DiD 设计必须提供 post_column、event_time_column，或完整的"
                    " time_column + treatment_start_column + unit_id_column。"
                )
            if bool(self.time_column) != bool(self.treatment_start_column):
                raise ValueError("错位处理设计必须同时提供 time_column 和 treatment_start_column。")
            if self.estimand != "ATT":
                raise ValueError("DiD 当前只支持 ATT estimand。")
        if self.design_type == "instrumental_variable":
            if not self.variable_roles.instruments:
                raise ValueError("工具变量设计必须明确提供 instruments。")
            if self.estimand != "LATE":
                raise ValueError("工具变量设计当前只支持 LATE estimand。")
        if self.design_type == "regression_discontinuity":
            if not self.running_variable:
                raise ValueError("RDD 设计必须明确提供 running_variable。")
            if self.cutoff is None:
                raise ValueError("RDD 设计必须明确提供 cutoff。")
            if self.bandwidth is not None and self.bandwidth <= 0:
                raise ValueError("RDD bandwidth 必须大于 0。")
            if self.estimand != "LATE":
                raise ValueError("RDD 当前只支持 cutoff 处的 LATE estimand。")
        if self.design_type == "synthetic_control":
            if not self.unit_id_column or not self.time_column:
                raise ValueError(
                    "合成控制必须明确提供 unit_id_column 和 time_column。"
                )
            if self.intervention_time is None:
                raise ValueError("合成控制必须明确提供 intervention_time。")
            if self.estimand != "ATT":
                raise ValueError("合成控制当前只支持 ATT estimand。")
        if self.design_type == "time_series":
            if not self.time_column:
                raise ValueError("时间序列设计必须明确提供 time_column。")
            if self.estimand != "IRF":
                raise ValueError("时间序列局部投影当前只支持 IRF estimand。")
            if not 0 <= self.max_horizon <= 60:
                raise ValueError("max_horizon 必须在 0 到 60 之间。")
            if self.treatment_lags < 1 or self.outcome_lags < 1:
                raise ValueError("treatment_lags 和 outcome_lags 必须至少为 1。")
            if self.control_lag < 1:
                raise ValueError("control_lag 必须至少为 1。")
            if self.hac_lags is not None and self.hac_lags < 1:
                raise ValueError("hac_lags 必须至少为 1。")
        for role, values in self.variable_roles.model_dump().items():
            overlap = sorted(reserved & set(values))
            if overlap:
                raise ValueError(
                    "Treatment、Outcome、时间或设计专属字段不能同时被标记为 "
                    f"{role}: {overlap}"
                )

        for field_name in (
            "target_population",
            "unit_of_analysis",
            "outcome_horizon",
        ):
            value = str(getattr(self, field_name)).strip()
            if not value:
                raise ValueError(f"{field_name} 不能为空。")
            setattr(self, field_name, value)

        return self


class CausalQuestionDraft(BaseModel):
    """
    从用户原始问题提取出的因果问题草案。

    与 CausalQuestion 不同，研究语境字段允许暂时缺失；缺失状态会明确
    保存在工作流中，不能由模型静默猜测。
    """

    treatment: str
    outcome: str
    estimand: Estimand = "ATE"
    design_type: StudyDesignType = "cross_sectional"
    post_column: str | None = None
    event_time_column: str | None = None
    reference_period: int = -1
    time_column: str | None = None
    treatment_start_column: str | None = None
    unit_id_column: str | None = None
    running_variable: str | None = None
    cutoff: float | None = None
    treatment_above_cutoff: bool = True
    bandwidth: float | None = None
    intervention_time: float | None = None
    treated_unit: str | int | float | None = None
    max_horizon: int = 5
    treatment_lags: int = 1
    outcome_lags: int = 1
    control_lag: int = 1
    cumulative_outcome: bool = True
    hac_lags: int | None = None
    treatment_contrast: TreatmentContrast = Field(
        default_factory=lambda: TreatmentContrast(
            control=0,
            treated=1,
        )
    )
    target_population: str | None = None
    unit_of_analysis: str | None = None
    outcome_horizon: str | None = None
    variable_roles: VariableRoles = Field(default_factory=VariableRoles)

    @model_validator(mode="after")
    def validate_draft(self) -> "CausalQuestionDraft":
        self.treatment = self.treatment.strip()
        self.outcome = self.outcome.strip()
        if not self.treatment or not self.outcome:
            raise ValueError("treatment 和 outcome 不能为空。")
        if self.treatment == self.outcome:
            raise ValueError(
                "treatment 和 outcome 不能是同一字段。"
            )

        for field_name in (
            "post_column",
            "event_time_column",
            "time_column",
            "treatment_start_column",
            "unit_id_column",
            "running_variable",
        ):
            value = getattr(self, field_name)
            if isinstance(value, str):
                normalized = value.strip()
                setattr(self, field_name, normalized or None)

        reserved = {self.treatment, self.outcome}
        if self.post_column:
            reserved.add(self.post_column)
        if self.event_time_column:
            reserved.add(self.event_time_column)
        if self.time_column:
            reserved.add(self.time_column)
        if self.treatment_start_column:
            reserved.add(self.treatment_start_column)
        if self.unit_id_column:
            reserved.add(self.unit_id_column)
        if self.running_variable:
            reserved.add(self.running_variable)
        if self.bandwidth is not None and self.bandwidth <= 0:
            raise ValueError("RDD bandwidth 必须大于 0。")
        if not 0 <= self.max_horizon <= 60:
            raise ValueError("max_horizon 必须在 0 到 60 之间。")
        if self.treatment_lags < 1 or self.outcome_lags < 1:
            raise ValueError("treatment_lags 和 outcome_lags 必须至少为 1。")
        if self.control_lag < 1:
            raise ValueError("control_lag 必须至少为 1。")
        if self.hac_lags is not None and self.hac_lags < 1:
            raise ValueError("hac_lags 必须至少为 1。")
        for role, values in self.variable_roles.model_dump().items():
            overlap = sorted(reserved & set(values))
            if overlap:
                raise ValueError(
                    "Treatment、Outcome、时间或设计专属字段不能同时被标记为 "
                    f"{role}: {overlap}"
                )

        for field_name in (
            "target_population",
            "unit_of_analysis",
            "outcome_horizon",
        ):
            value = getattr(self, field_name)
            if isinstance(value, str):
                normalized = value.strip()
                setattr(self, field_name, normalized or None)

        return self

    @property
    def missing_design_fields(self) -> list[str]:
        return [
            field_name
            for field_name in (
                "target_population",
                "unit_of_analysis",
                "outcome_horizon",
            )
            if getattr(self, field_name) is None
        ] + (
            ["post_column_or_event_time_column"]
            if self.design_type == "difference_in_differences"
            and self.post_column is None
            and self.event_time_column is None
            and not (
                self.time_column
                and self.treatment_start_column
                and self.unit_id_column
            )
            else []
        ) + (
            ["instruments"]
            if self.design_type == "instrumental_variable"
            and not self.variable_roles.instruments
            else []
        ) + (
            ["running_variable"]
            if self.design_type == "regression_discontinuity"
            and self.running_variable is None
            else []
        ) + (
            ["cutoff"]
            if self.design_type == "regression_discontinuity"
            and self.cutoff is None
            else []
        ) + (
            ["unit_id_column"]
            if self.design_type == "synthetic_control"
            and self.unit_id_column is None
            else []
        ) + (
            ["time_column"]
            if self.design_type == "synthetic_control"
            and self.time_column is None
            else []
        ) + (
            ["intervention_time"]
            if self.design_type == "synthetic_control"
            and self.intervention_time is None
            else []
        ) + (
            ["time_column"]
            if self.design_type == "time_series" and self.time_column is None
            else []
        )

    def confirm(self) -> CausalQuestion:
        """仅在研究语境完整时转换为可执行的已确认问题。"""

        if self.missing_design_fields:
            raise ValueError(
                "因果问题仍缺少研究设计字段："
                + ", ".join(self.missing_design_fields)
            )

        return CausalQuestion(
            treatment=self.treatment,
            outcome=self.outcome,
            estimand=self.estimand,
            design_type=self.design_type,
            post_column=self.post_column,
            event_time_column=self.event_time_column,
            reference_period=self.reference_period,
            time_column=self.time_column,
            treatment_start_column=self.treatment_start_column,
            unit_id_column=self.unit_id_column,
            running_variable=self.running_variable,
            cutoff=self.cutoff,
            treatment_above_cutoff=self.treatment_above_cutoff,
            bandwidth=self.bandwidth,
            intervention_time=self.intervention_time,
            treated_unit=self.treated_unit,
            max_horizon=self.max_horizon,
            treatment_lags=self.treatment_lags,
            outcome_lags=self.outcome_lags,
            control_lag=self.control_lag,
            cumulative_outcome=self.cumulative_outcome,
            hac_lags=self.hac_lags,
            treatment_contrast=self.treatment_contrast,
            target_population=self.target_population or "",
            unit_of_analysis=self.unit_of_analysis or "",
            outcome_horizon=self.outcome_horizon or "",
            variable_roles=self.variable_roles,
        )


class StudyDesign(BaseModel):
    """可审计的因果研究设计快照。"""

    question: CausalQuestion
    dag_edges: list[tuple[str, str]] = Field(default_factory=list)
    adjustment_set: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    status: DesignStatus = "draft"
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_design(self) -> "StudyDesign":
        self.adjustment_set = [
            item.strip() for item in self.adjustment_set
        ]
        if len(self.adjustment_set) != len(set(self.adjustment_set)):
            raise ValueError("adjustment_set 不能包含重复字段。")

        for source, target in self.dag_edges:
            if not source.strip() or not target.strip():
                raise ValueError("DAG 边的端点不能为空。")
            if source == target:
                raise ValueError("DAG 不允许自环。")

        return self
