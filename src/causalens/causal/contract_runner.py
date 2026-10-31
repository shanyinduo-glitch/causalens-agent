from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import pandas as pd
from pydantic import ValidationError

from causalens.agent.causal_specification import (
    validate_causal_configuration_payload,
)
from causalens.causal.data_quality import assess_causal_data_quality
from causalens.causal.design_router import select_causal_design
from causalens.causal.diagnostics import run_causal_diagnostics
from causalens.causal.identification import assess_causal_identification
from causalens.domain.causal_question import CausalQuestion
from causalens.tools.causal_readiness import check_causal_readiness
from causalens.tools.causal_tool import run_causal_estimators
from causalens.tools.did_tool import check_did_readiness
from causalens.tools.event_study_tool import check_event_study_readiness
from causalens.tools.iv_tool import check_iv_readiness
from causalens.tools.rdd_tool import check_rdd_readiness
from causalens.tools.staggered_did_tool import check_staggered_did_readiness
from causalens.tools.synthetic_control_tool import (
    check_synthetic_control_readiness,
)
from causalens.tools.time_series_tool import check_time_series_readiness


def _result(
    *,
    workflow_status: str,
    stage: str,
    contract: dict[str, Any] | None = None,
    design_selection: dict[str, Any] | None = None,
    identification: dict[str, Any] | None = None,
    data_quality: dict[str, Any] | None = None,
    readiness: dict[str, Any] | None = None,
    diagnostics: dict[str, Any] | None = None,
    estimation: dict[str, Any] | None = None,
    errors: list[str] | None = None,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "success": workflow_status in {"awaiting_approval", "completed"},
        "workflow_status": workflow_status,
        "stage": stage,
        "contract": contract,
        "design_selection": design_selection,
        "identification": identification,
        "data_quality": data_quality,
        "readiness": readiness,
        "diagnostics": diagnostics,
        "estimation": estimation,
        "errors": list(errors or []),
        "warnings": list(dict.fromkeys(warnings or [])),
        "llm_calls": 0,
    }


def _design_assumptions(question: CausalQuestion) -> dict[str, Any]:
    warnings = {
        "difference_in_differences": (
            "DiD 依赖平行趋势、无提前反应及无组别特异同期冲击；"
            "这些假设不能由 CSV 自动证明。"
        ),
        "instrumental_variable": (
            "IV 依赖相关性、排除限制、独立性与单调性；其中排除限制不能由"
            "当前数据自动证明。"
        ),
        "regression_discontinuity": (
            "RDD 依赖阈值附近不可精确操纵及潜在结果连续性；这些条件需要"
            "领域证据支持。"
        ),
        "synthetic_control": (
            "合成控制依赖供体未受干预、干预前拟合充分且没有同期结构突变。"
        ),
        "time_series": (
            "时间序列局部投影依赖冲击在滞后控制后近似外生、动态设定充分、"
            "关系稳定且不存在遗漏的同期共同冲击；HAC 标准误不能证明这些条件。"
        ),
    }
    return {
        "status": "not_assessed",
        "identified": None,
        "dag_provided": False,
        "dag_valid": None,
        "treatment": question.treatment,
        "outcome": question.outcome,
        "adjustment_set": question.variable_roles.confounders,
        "dag_edges": [],
        "blocking_issues": [],
        "warnings": [warnings[question.design_type]],
        "diagnostics": {"design_type": question.design_type},
    }


def _quality_columns(question: CausalQuestion) -> tuple[list[str], list[str]]:
    roles = question.variable_roles
    required = list(
        dict.fromkeys(
            [
                question.treatment,
                question.outcome,
                *roles.confounders,
                *roles.effect_modifiers,
                *roles.instruments,
                *roles.negative_control_outcomes,
                *[
                    column
                    for column in (
                        question.post_column,
                        question.event_time_column,
                        question.time_column,
                        question.treatment_start_column,
                        question.unit_id_column,
                        question.running_variable,
                    )
                    if column
                ],
            ]
        )
    )
    non_numeric = {question.unit_id_column}
    if question.design_type == "time_series":
        non_numeric.add(question.time_column)
    numeric = [column for column in required if column not in non_numeric]
    return required, numeric


def _readiness_check(
    question: CausalQuestion,
    dataset_path: str,
) -> tuple[str, Callable[..., dict[str, Any]], dict[str, Any]]:
    roles = question.variable_roles
    common = {
        "dataset_path": dataset_path,
        "outcome": question.outcome,
    }
    if question.design_type == "synthetic_control":
        return (
            "check_synthetic_control_readiness",
            check_synthetic_control_readiness,
            {
                **common,
                "treatment": question.treatment,
                "unit_id_column": question.unit_id_column,
                "time_column": question.time_column,
                "intervention_time": question.intervention_time,
                "covariates": roles.confounders,
                "treated_unit": question.treated_unit,
            },
        )
    if question.design_type == "time_series":
        return (
            "check_time_series_readiness",
            check_time_series_readiness,
            {
                **common,
                "treatment": question.treatment,
                "time_column": question.time_column,
                "confounders": roles.confounders,
                "max_horizon": question.max_horizon,
                "treatment_lags": question.treatment_lags,
                "outcome_lags": question.outcome_lags,
            },
        )
    if question.design_type == "regression_discontinuity":
        return (
            "check_rdd_readiness",
            check_rdd_readiness,
            {
                **common,
                "treatment": question.treatment,
                "running_variable": question.running_variable,
                "cutoff": question.cutoff,
                "treatment_above_cutoff": question.treatment_above_cutoff,
                "covariates": roles.confounders,
                "bandwidth": question.bandwidth,
            },
        )
    if question.design_type == "instrumental_variable":
        return (
            "check_iv_readiness",
            check_iv_readiness,
            {
                **common,
                "treatment": question.treatment,
                "instruments": roles.instruments,
                "confounders": roles.confounders,
            },
        )
    if question.design_type == "difference_in_differences":
        if (
            question.time_column
            and question.treatment_start_column
            and question.unit_id_column
        ):
            return (
                "check_staggered_did_readiness",
                check_staggered_did_readiness,
                {
                    **common,
                    "treatment_column": question.treatment,
                    "time_column": question.time_column,
                    "treatment_start_column": question.treatment_start_column,
                    "unit_id_column": question.unit_id_column,
                    "confounders": roles.confounders,
                },
            )
        if question.event_time_column:
            return (
                "check_event_study_readiness",
                check_event_study_readiness,
                {
                    **common,
                    "group_column": question.treatment,
                    "event_time_column": question.event_time_column,
                    "confounders": roles.confounders,
                    "unit_id_column": question.unit_id_column,
                    "post_column": question.post_column,
                    "reference_period": question.reference_period,
                },
            )
        return (
            "check_did_readiness",
            check_did_readiness,
            {
                **common,
                "group_column": question.treatment,
                "post_column": question.post_column,
                "confounders": roles.confounders,
                "unit_id_column": question.unit_id_column,
            },
        )
    return (
        "check_causal_readiness",
        check_causal_readiness,
        {
            **common,
            "treatment": question.treatment,
            "confounders": list(
                dict.fromkeys([*roles.confounders, *roles.effect_modifiers])
            ),
        },
    )


def run_explicit_causal_contract(
    *,
    dataset_path: str | Path,
    contract: dict[str, Any],
    approved: bool = False,
    result_cache_dir: str | Path | None = None,
    chart_output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Validate and optionally execute a causal contract without any LLM call."""

    path = Path(dataset_path)
    if not path.is_file():
        return _result(
            workflow_status="failed",
            stage="dataset_validation",
            errors=[f"Dataset file does not exist: {path}"],
        )
    if not isinstance(contract, dict):
        return _result(
            workflow_status="failed",
            stage="contract_validation",
            errors=["Causal contract must be a JSON object."],
        )
    try:
        columns = set(pd.read_csv(path, nrows=0).columns)
    except Exception as exc:
        return _result(
            workflow_status="failed",
            stage="dataset_validation",
            errors=[f"Failed to read CSV header: {exc}"],
        )

    configuration, errors = validate_causal_configuration_payload(
        payload=contract,
        available_columns=columns,
    )
    if errors or configuration is None:
        return _result(
            workflow_status="blocked",
            stage="contract_validation",
            errors=errors or ["Causal contract validation failed."],
        )
    missing = list(configuration.get("missing_design_fields") or [])
    if missing:
        return _result(
            workflow_status="blocked",
            stage="contract_validation",
            contract=configuration["causal_question"],
            errors=[f"研究契约仍缺少字段：{missing}"],
        )
    try:
        question = CausalQuestion.model_validate(configuration["causal_question"])
    except ValidationError as exc:
        return _result(
            workflow_status="blocked",
            stage="contract_validation",
            contract=configuration["causal_question"],
            errors=[str(exc)],
        )

    question_payload = question.model_dump()
    roles = question.variable_roles
    design = select_causal_design(
        design_type=question.design_type,
        estimand=question.estimand,
        post_column=question.post_column,
        event_time_column=question.event_time_column,
        time_column=question.time_column,
        treatment_start_column=question.treatment_start_column,
        unit_id_column=question.unit_id_column,
        running_variable=question.running_variable,
        cutoff=question.cutoff,
        intervention_time=question.intervention_time,
        instruments=roles.instruments,
        effect_modifiers=roles.effect_modifiers,
    )
    if design["ready"] is not True:
        return _result(
            workflow_status="blocked",
            stage="design_selection",
            contract=question_payload,
            design_selection=design,
            errors=list(design["blocking_issues"]),
        )

    dag_edges = [tuple(edge) for edge in configuration["causal_dag_edges"]]
    if question.design_type == "cross_sectional":
        identification_model = assess_causal_identification(
            treatment=question.treatment,
            outcome=question.outcome,
            adjustment_set=roles.confounders,
            dag_edges=dag_edges,
        )
        identification = identification_model.model_dump()
    else:
        identification = _design_assumptions(question)
    if identification["status"] == "blocked":
        return _result(
            workflow_status="blocked",
            stage="identification",
            contract=question_payload,
            design_selection=design,
            identification=identification,
            errors=list(identification["blocking_issues"]),
            warnings=list(identification["warnings"]),
        )

    required, numeric = _quality_columns(question)
    quality_response = assess_causal_data_quality(
        dataset_path=str(path),
        required_columns=required,
        numeric_columns=numeric,
        adjustment_columns=list(
            dict.fromkeys([*roles.confounders, *roles.effect_modifiers])
        ),
        panel_key=(question.unit_id_column, question.time_column)
        if question.unit_id_column and question.time_column
        else None,
    )
    if quality_response.get("success") is not True:
        return _result(
            workflow_status="failed",
            stage="data_quality",
            contract=question_payload,
            design_selection=design,
            identification=identification,
            errors=[str(quality_response.get("error") or "Data-quality check failed.")],
        )
    quality = quality_response["result"]
    if quality["ready"] is not True:
        return _result(
            workflow_status="blocked",
            stage="data_quality",
            contract=question_payload,
            design_selection=design,
            identification=identification,
            data_quality=quality,
            errors=list(quality["blocking_issues"]),
            warnings=list(quality["warnings"]),
        )

    readiness_name, readiness_callable, readiness_arguments = _readiness_check(
        question, str(path)
    )
    try:
        readiness_response = readiness_callable(**readiness_arguments)
    except Exception as exc:
        return _result(
            workflow_status="failed",
            stage="readiness",
            contract=question_payload,
            design_selection=design,
            identification=identification,
            data_quality=quality,
            errors=[f"{readiness_name} raised an exception: {exc}"],
        )
    if readiness_response.get("success") is not True:
        return _result(
            workflow_status="failed",
            stage="readiness",
            contract=question_payload,
            design_selection=design,
            identification=identification,
            data_quality=quality,
            errors=[str(readiness_response.get("error") or "Readiness check failed.")],
        )
    readiness = readiness_response["result"]
    if readiness.get("ready") is not True:
        return _result(
            workflow_status="blocked",
            stage="readiness",
            contract=question_payload,
            design_selection=design,
            identification=identification,
            data_quality=quality,
            readiness={"tool_name": readiness_name, **readiness},
            errors=list(readiness.get("blocking_issues") or []),
            warnings=[
                *identification["warnings"],
                *quality["warnings"],
                *(readiness.get("warnings") or []),
            ],
        )

    diagnostics: dict[str, Any] | None = None
    if question.design_type == "cross_sectional":
        diagnostics_response = run_causal_diagnostics(
            dataset_path=str(path),
            treatment=question.treatment,
            outcome=question.outcome,
            confounders=list(
                dict.fromkeys([*roles.confounders, *roles.effect_modifiers])
            ),
        )
        if diagnostics_response.get("success") is not True:
            return _result(
                workflow_status="failed",
                stage="diagnostics",
                contract=question_payload,
                design_selection=design,
                identification=identification,
                data_quality=quality,
                readiness={"tool_name": readiness_name, **readiness},
                errors=[str(diagnostics_response.get("error") or "Diagnostics failed.")],
            )
        diagnostics = diagnostics_response["result"]
        if diagnostics.get("passed") is not True:
            return _result(
                workflow_status="blocked",
                stage="diagnostics",
                contract=question_payload,
                design_selection=design,
                identification=identification,
                data_quality=quality,
                readiness={"tool_name": readiness_name, **readiness},
                diagnostics=diagnostics,
                errors=list(diagnostics.get("blocking_issues") or []),
                warnings=list(diagnostics.get("warnings") or []),
            )

    accumulated_warnings = [
        *identification["warnings"],
        *quality["warnings"],
        *(readiness.get("warnings") or []),
        *((diagnostics or {}).get("warnings") or []),
    ]
    readiness_payload = {"tool_name": readiness_name, **readiness}
    if not approved:
        return _result(
            workflow_status="awaiting_approval",
            stage="approval",
            contract=question_payload,
            design_selection=design,
            identification=identification,
            data_quality=quality,
            readiness=readiness_payload,
            diagnostics=diagnostics,
            warnings=accumulated_warnings,
        )

    estimation = run_causal_estimators(
        dataset_path=str(path),
        treatment=question.treatment,
        outcome=question.outcome,
        confounders=roles.confounders,
        effect_modifiers=roles.effect_modifiers,
        negative_control_outcomes=roles.negative_control_outcomes,
        instruments=roles.instruments,
        estimand=question.estimand,
        design_type=question.design_type,
        post_column=question.post_column,
        event_time_column=question.event_time_column,
        reference_period=question.reference_period,
        time_column=question.time_column,
        treatment_start_column=question.treatment_start_column,
        unit_id_column=question.unit_id_column,
        running_variable=question.running_variable,
        cutoff=question.cutoff,
        treatment_above_cutoff=question.treatment_above_cutoff,
        bandwidth=question.bandwidth,
        intervention_time=question.intervention_time,
        treated_unit=question.treated_unit,
        max_horizon=question.max_horizon,
        treatment_lags=question.treatment_lags,
        outcome_lags=question.outcome_lags,
        control_lag=question.control_lag,
        cumulative_outcome=question.cumulative_outcome,
        hac_lags=question.hac_lags,
        chart_output_dir=str(chart_output_dir) if chart_output_dir else None,
        result_cache_dir=str(result_cache_dir) if result_cache_dir else None,
    )
    if estimation.get("success") is not True:
        return _result(
            workflow_status="failed",
            stage="estimation",
            contract=question_payload,
            design_selection=design,
            identification=identification,
            data_quality=quality,
            readiness=readiness_payload,
            diagnostics=diagnostics,
            estimation=estimation,
            errors=[str(estimation.get("error") or "Causal estimation failed.")],
            warnings=accumulated_warnings,
        )
    return _result(
        workflow_status="completed",
        stage="completed",
        contract=question_payload,
        design_selection=design,
        identification=identification,
        data_quality=quality,
        readiness=readiness_payload,
        diagnostics=diagnostics,
        estimation=estimation,
        warnings=accumulated_warnings,
    )
