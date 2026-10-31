from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from causalens.domain import CausalQuestionDraft, VariableRoles


CAUSAL_CONFIGURATION_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "extract_causal_configuration",
            "description": (
                "Extract only explicitly stated causal study fields. "
                "Never infer missing study design information."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "treatment": {
                        "type": "string",
                        "description": (
                            "Explicit treatment column, or an empty string."
                        ),
                    },
                    "outcome": {
                        "type": "string",
                        "description": (
                            "Explicit outcome column, or an empty string."
                        ),
                    },
                    "confounders": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Only explicitly stated confounder columns."
                        ),
                    },
                    "estimand": {
                        "type": "string",
                        "enum": ["ATE", "ATT", "ATC", "CATE", "LATE", "IRF"],
                        "description": (
                            "Explicit estimand; use ATE when not stated."
                        ),
                    },
                    "design_type": {
                        "type": "string",
                        "enum": [
                            "cross_sectional",
                            "difference_in_differences",
                            "instrumental_variable",
                            "regression_discontinuity",
                            "synthetic_control",
                            "time_series",
                        ],
                    },
                    "running_variable": {
                        "type": "string",
                        "description": (
                            "Explicit numeric RDD running/forcing variable, or empty."
                        ),
                    },
                    "cutoff": {
                        "type": "number",
                        "description": "Explicit RDD assignment cutoff.",
                    },
                    "treatment_above_cutoff": {
                        "type": "boolean",
                        "description": (
                            "True when units at/above cutoff receive treatment; "
                            "false when units below cutoff receive treatment."
                        ),
                    },
                    "bandwidth": {
                        "type": "number",
                        "description": (
                            "Optional positive RDD bandwidth; omit for local rule."
                        ),
                    },
                    "intervention_time": {
                        "type": "number",
                        "description": (
                            "Explicit first post-treatment time for synthetic control."
                        ),
                    },
                    "treated_unit": {
                        "description": (
                            "Optional explicit treated unit ID; omit to infer it from "
                            "the binary treatment column."
                        ),
                        "oneOf": [{"type": "string"}, {"type": "number"}],
                    },
                    "post_column": {
                        "type": "string",
                        "description": "Explicit 0/1 post-period column, or empty.",
                    },
                    "event_time_column": {
                        "type": "string",
                        "description": (
                            "Explicit integer relative-event-time column for "
                            "a multi-period Event Study, or empty."
                        ),
                    },
                    "reference_period": {
                        "type": "integer",
                        "description": (
                            "Omitted relative period used as the Event Study "
                            "reference; use -1 when not explicitly stated."
                        ),
                    },
                    "time_column": {
                        "type": "string",
                        "description": (
                            "Explicit integer calendar-period column for a "
                            "staggered-adoption DiD design, or empty."
                        ),
                    },
                    "max_horizon": {
                        "type": "integer",
                        "description": "Maximum local-projection response horizon.",
                    },
                    "treatment_lags": {
                        "type": "integer",
                        "description": "Positive number of treatment lags.",
                    },
                    "outcome_lags": {
                        "type": "integer",
                        "description": "Positive number of outcome lags.",
                    },
                    "control_lag": {
                        "type": "integer",
                        "description": "Positive lag applied to declared controls.",
                    },
                    "cumulative_outcome": {
                        "type": "boolean",
                        "description": "Whether response through horizon h is cumulative.",
                    },
                    "hac_lags": {
                        "type": "integer",
                        "description": "Optional positive Newey-West maximum lag.",
                    },
                    "treatment_start_column": {
                        "type": "string",
                        "description": (
                            "Explicit unit-level first-treatment-period column; "
                            "never-treated units must be empty."
                        ),
                    },
                    "unit_id_column": {
                        "type": "string",
                        "description": (
                            "Explicit stable panel unit ID, or empty for a "
                            "repeated cross-section DiD design."
                        ),
                    },
                    "target_population": {
                        "type": "string",
                        "description": (
                            "Explicit target population, or empty string."
                        ),
                    },
                    "unit_of_analysis": {
                        "type": "string",
                        "description": (
                            "Explicit unit of analysis, or empty string."
                        ),
                    },
                    "outcome_horizon": {
                        "type": "string",
                        "description": (
                            "Explicit outcome time horizon, or empty string."
                        ),
                    },
                    "effect_modifiers": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "mediators": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "colliders": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "instruments": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "negative_control_outcomes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Only outcome columns explicitly declared to be "
                            "unaffected by treatment over the study horizon."
                        ),
                    },
                    "dag_edges": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "source": {"type": "string"},
                                "target": {"type": "string"},
                            },
                            "required": ["source", "target"],
                            "additionalProperties": False,
                        },
                        "description": (
                            "Only directed DAG edges explicitly supplied by "
                            "the user; otherwise return an empty list."
                        ),
                    },
                },
                "required": ["treatment", "outcome", "confounders"],
                "additionalProperties": False,
            },
        },
    },
]


ROLE_FIELDS = (
    "confounders",
    "effect_modifiers",
    "mediators",
    "colliders",
    "instruments",
    "negative_control_outcomes",
)


def _normalize_optional_text(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("必须是字符串。")
    normalized = value.strip()
    return normalized or None


def _normalize_role_list(
    payload: dict[str, Any],
    field_name: str,
    *,
    required: bool = False,
) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    value = payload.get(field_name, [])
    if not isinstance(value, list):
        return [], [f"{field_name} 必须是列表。"]
    if not all(isinstance(item, str) and item.strip() for item in value):
        return [], [f"{field_name} 中的字段名必须是非空字符串。"]

    normalized = [item.strip() for item in value]
    if len(normalized) != len(set(normalized)):
        errors.append(f"{field_name} 中存在重复字段。")
    if required and not normalized:
        errors.append(f"用户没有明确指定 {field_name}。")
    return normalized, errors


def _normalize_dag_edges(
    value: Any,
) -> tuple[list[tuple[str, str]], list[str]]:
    if value is None:
        return [], []
    if not isinstance(value, list):
        return [], ["dag_edges 必须是列表。"]

    edges: list[tuple[str, str]] = []
    errors: list[str] = []
    for index, edge in enumerate(value):
        if not isinstance(edge, dict):
            errors.append(f"dag_edges[{index}] 必须是对象。")
            continue
        source = edge.get("source")
        target = edge.get("target")
        if not isinstance(source, str) or not source.strip():
            errors.append(f"dag_edges[{index}].source 必须是非空字符串。")
            continue
        if not isinstance(target, str) or not target.strip():
            errors.append(f"dag_edges[{index}].target 必须是非空字符串。")
            continue
        edges.append((source.strip(), target.strip()))

    if len(edges) != len(set(edges)):
        errors.append("dag_edges 中存在重复边。")
    return edges, errors


def validate_causal_configuration_payload(
    payload: dict[str, Any],
    available_columns: set[str],
) -> tuple[dict[str, Any] | None, list[str]]:
    """验证模型提取结果，不猜测字段、不静默修改变量角色。"""

    errors: list[str] = []
    treatment = payload.get("treatment")
    outcome = payload.get("outcome")

    if not isinstance(treatment, str):
        errors.append("treatment 必须是字符串。")
        treatment = ""
    else:
        treatment = treatment.strip()

    if not isinstance(outcome, str):
        errors.append("outcome 必须是字符串。")
        outcome = ""
    else:
        outcome = outcome.strip()

    if not treatment:
        errors.append("用户没有明确指定 treatment。")
    if not outcome:
        errors.append("用户没有明确指定 outcome。")
    if treatment and outcome and treatment == outcome:
        errors.append("treatment 和 outcome 不能是同一字段。")

    raw_design_type = payload.get("design_type") or "cross_sectional"
    roles: dict[str, list[str]] = {}
    for field_name in ROLE_FIELDS:
        values, role_errors = _normalize_role_list(
            payload,
            field_name,
            required=(
                field_name == "confounders"
                and raw_design_type == "cross_sectional"
            ),
        )
        roles[field_name] = values
        errors.extend(role_errors)

    dag_edges, dag_errors = _normalize_dag_edges(
        payload.get("dag_edges")
    )
    errors.extend(dag_errors)

    estimand = payload.get("estimand") or "ATE"
    if estimand not in {"ATE", "ATT", "ATC", "CATE", "LATE", "IRF"}:
        errors.append("estimand 必须是 ATE、ATT、ATC、CATE、LATE 或 IRF。")

    design_type = payload.get("design_type") or "cross_sectional"
    if design_type not in {
        "cross_sectional",
        "difference_in_differences",
        "instrumental_variable",
        "regression_discontinuity",
        "synthetic_control",
        "time_series",
    }:
        errors.append(
            "design_type 必须是 cross_sectional、difference_in_differences "
            "instrumental_variable、regression_discontinuity、synthetic_control 或 "
            "time_series。"
        )

    design_columns: dict[str, str | None] = {}
    for field_name in (
        "post_column",
        "event_time_column",
        "time_column",
        "treatment_start_column",
        "unit_id_column",
    ):
        try:
            design_columns[field_name] = _normalize_optional_text(
                payload.get(field_name)
            )
        except ValueError:
            design_columns[field_name] = None
            errors.append(f"{field_name} 必须是字符串。")
    if design_type == "difference_in_differences":
        has_staggered_contract = bool(
            design_columns["time_column"]
            and design_columns["treatment_start_column"]
            and design_columns["unit_id_column"]
        )
        if (
            not design_columns["post_column"]
            and not design_columns["event_time_column"]
            and not has_staggered_contract
        ):
            errors.append(
                "DiD 请求必须明确提供 post_column、event_time_column，或完整的"
                " time_column + treatment_start_column + unit_id_column。"
            )
        if bool(design_columns["time_column"]) != bool(
            design_columns["treatment_start_column"]
        ):
            errors.append(
                "错位处理设计必须同时提供 time_column 和 treatment_start_column。"
            )
        if estimand != "ATT":
            errors.append("DiD 当前只支持 ATT estimand。")
    if design_type == "instrumental_variable":
        if not roles["instruments"]:
            errors.append("工具变量设计必须明确提供至少一个 instrument。")
        if estimand != "LATE":
            errors.append("工具变量设计当前只支持 LATE estimand。")
    running_variable: str | None
    try:
        running_variable = _normalize_optional_text(
            payload.get("running_variable")
        )
    except ValueError:
        running_variable = None
        errors.append("running_variable 必须是字符串。")
    cutoff = payload.get("cutoff")
    if isinstance(cutoff, bool) or (
        cutoff is not None and not isinstance(cutoff, (int, float))
    ):
        errors.append("cutoff 必须是数值。")
        cutoff = None
    bandwidth = payload.get("bandwidth")
    if isinstance(bandwidth, bool) or (
        bandwidth is not None and not isinstance(bandwidth, (int, float))
    ):
        errors.append("bandwidth 必须是正数。")
        bandwidth = None
    elif bandwidth is not None and bandwidth <= 0:
        errors.append("bandwidth 必须大于 0。")
    treatment_above_cutoff = payload.get("treatment_above_cutoff", True)
    if not isinstance(treatment_above_cutoff, bool):
        errors.append("treatment_above_cutoff 必须是布尔值。")
        treatment_above_cutoff = True
    if design_type == "regression_discontinuity":
        if not running_variable:
            errors.append("RDD 设计必须明确提供 running_variable。")
        if cutoff is None:
            errors.append("RDD 设计必须明确提供 cutoff。")
        if estimand != "LATE":
            errors.append("RDD 当前只支持 cutoff 处的 LATE estimand。")
    intervention_time = payload.get("intervention_time")
    if isinstance(intervention_time, bool) or (
        intervention_time is not None
        and not isinstance(intervention_time, (int, float))
    ):
        errors.append("intervention_time 必须是数值。")
        intervention_time = None
    treated_unit = payload.get("treated_unit")
    if isinstance(treated_unit, bool) or (
        treated_unit is not None
        and not isinstance(treated_unit, (str, int, float))
    ):
        errors.append("treated_unit 必须是字符串或数值。")
        treated_unit = None
    if design_type == "synthetic_control":
        if not design_columns["unit_id_column"]:
            errors.append("合成控制必须明确提供 unit_id_column。")
        if not design_columns["time_column"]:
            errors.append("合成控制必须明确提供 time_column。")
        if intervention_time is None:
            errors.append("合成控制必须明确提供 intervention_time。")
        if estimand != "ATT":
            errors.append("合成控制当前只支持 ATT estimand。")

    integer_defaults = {
        "max_horizon": 5,
        "treatment_lags": 1,
        "outcome_lags": 1,
        "control_lag": 1,
    }
    time_series_parameters: dict[str, int | bool | None] = {}
    for field_name, default in integer_defaults.items():
        value = payload.get(field_name, default)
        if isinstance(value, bool) or not isinstance(value, int):
            errors.append(f"{field_name} 必须是整数。")
            value = default
        time_series_parameters[field_name] = value
    if not 0 <= int(time_series_parameters["max_horizon"] or 0) <= 60:
        errors.append("max_horizon 必须在 0 到 60 之间。")
    for field_name in ("treatment_lags", "outcome_lags", "control_lag"):
        if int(time_series_parameters[field_name] or 0) < 1:
            errors.append(f"{field_name} 必须至少为 1。")
    cumulative_outcome = payload.get("cumulative_outcome", True)
    if not isinstance(cumulative_outcome, bool):
        errors.append("cumulative_outcome 必须是布尔值。")
        cumulative_outcome = True
    time_series_parameters["cumulative_outcome"] = cumulative_outcome
    hac_lags = payload.get("hac_lags")
    if isinstance(hac_lags, bool) or (
        hac_lags is not None and not isinstance(hac_lags, int)
    ):
        errors.append("hac_lags 必须是正整数。")
        hac_lags = None
    elif hac_lags is not None and hac_lags < 1:
        errors.append("hac_lags 必须至少为 1。")
    time_series_parameters["hac_lags"] = hac_lags
    if design_type == "time_series":
        if not design_columns["time_column"]:
            errors.append("时间序列设计必须明确提供 time_column。")
        if estimand != "IRF":
            errors.append("时间序列局部投影当前只支持 IRF estimand。")

    reference_period = payload.get("reference_period", -1)
    if isinstance(reference_period, bool) or not isinstance(reference_period, int):
        errors.append("reference_period 必须是整数。")
        reference_period = -1

    requested_columns = {
        treatment,
        outcome,
        *(item for values in roles.values() for item in values),
        *(item for edge in dag_edges for item in edge),
        *(item for item in design_columns.values() if item),
        *(item for item in [running_variable] if item),
    }
    requested_columns.discard("")
    missing_columns = sorted(requested_columns - available_columns)
    if missing_columns:
        errors.append(
            "以下因果分析字段不存在于 CSV："
            f"{missing_columns}。可用字段：{sorted(available_columns)}"
        )

    optional_text: dict[str, str | None] = {}
    for field_name in (
        "target_population",
        "unit_of_analysis",
        "outcome_horizon",
    ):
        try:
            optional_text[field_name] = _normalize_optional_text(
                payload.get(field_name)
            )
        except ValueError:
            optional_text[field_name] = None
            errors.append(f"{field_name} 必须是字符串。")

    if errors:
        return None, errors

    try:
        draft = CausalQuestionDraft(
            treatment=treatment,
            outcome=outcome,
            estimand=estimand,
            design_type=design_type,
            post_column=design_columns["post_column"],
            event_time_column=design_columns["event_time_column"],
            reference_period=reference_period,
            time_column=design_columns["time_column"],
            treatment_start_column=design_columns["treatment_start_column"],
            unit_id_column=design_columns["unit_id_column"],
            running_variable=running_variable,
            cutoff=float(cutoff) if cutoff is not None else None,
            treatment_above_cutoff=treatment_above_cutoff,
            bandwidth=float(bandwidth) if bandwidth is not None else None,
            intervention_time=(
                float(intervention_time)
                if intervention_time is not None
                else None
            ),
            treated_unit=treated_unit,
            max_horizon=int(time_series_parameters["max_horizon"] or 0),
            treatment_lags=int(time_series_parameters["treatment_lags"] or 1),
            outcome_lags=int(time_series_parameters["outcome_lags"] or 1),
            control_lag=int(time_series_parameters["control_lag"] or 1),
            cumulative_outcome=bool(time_series_parameters["cumulative_outcome"]),
            hac_lags=(
                int(time_series_parameters["hac_lags"])
                if time_series_parameters["hac_lags"] is not None
                else None
            ),
            target_population=optional_text["target_population"],
            unit_of_analysis=optional_text["unit_of_analysis"],
            outcome_horizon=optional_text["outcome_horizon"],
            variable_roles=VariableRoles(**roles),
        )
    except ValidationError as exc:
        return None, [
            "因果变量角色配置无效："
            + "; ".join(
                str(item["msg"]) for item in exc.errors()
            )
        ]

    return {
        "treatment": draft.treatment,
        "outcome": draft.outcome,
        "confounders": draft.variable_roles.confounders,
        "causal_question": draft.model_dump(),
        "causal_dag_edges": dag_edges,
        "missing_design_fields": draft.missing_design_fields,
    }, []


def build_causal_question_draft(
    configuration: dict[str, Any],
) -> CausalQuestionDraft:
    """兼容旧配置或新配置，构建统一的因果问题草案。"""

    embedded = configuration.get("causal_question")
    if isinstance(embedded, dict):
        return CausalQuestionDraft.model_validate(embedded)

    return CausalQuestionDraft(
        treatment=configuration["treatment"],
        outcome=configuration["outcome"],
        variable_roles=VariableRoles(
            confounders=list(configuration.get("confounders") or []),
        ),
    )
