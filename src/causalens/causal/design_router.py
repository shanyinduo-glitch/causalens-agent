from __future__ import annotations

from typing import Any

from causalens.causal.estimators import select_estimators


def select_causal_design(
    *,
    design_type: str,
    estimand: str,
    treatment_type: str = "binary",
    post_column: str | None = None,
    event_time_column: str | None = None,
    time_column: str | None = None,
    treatment_start_column: str | None = None,
    unit_id_column: str | None = None,
    running_variable: str | None = None,
    cutoff: float | None = None,
    intervention_time: float | None = None,
    instruments: list[str] | None = None,
    effect_modifiers: list[str] | None = None,
) -> dict[str, Any]:
    """将显式研究契约映射到注册估计器，不让 LLM 自由猜测方法。"""

    blocking: list[str] = []
    rationale: list[str] = []
    data_structure = "cross_sectional"
    study_design = ""
    if design_type == "cross_sectional":
        if estimand not in {"ATE", "CATE"}:
            blocking.append("Cross-sectional adjustment supports ATE or CATE only.")
        if estimand == "CATE" and not effect_modifiers:
            blocking.append("CATE requires explicit pre-treatment effect modifiers.")
        study_design = "cross_sectional_adjustment"
        rationale.append("Observed-confounder adjustment on cross-sectional data.")
    elif design_type == "difference_in_differences":
        if estimand != "ATT":
            blocking.append("Difference-in-Differences supports ATT only.")
        if time_column and treatment_start_column and unit_id_column:
            data_structure = "panel"
            study_design = "difference_in_differences_staggered_adoption"
            rationale.append("Unit panel with staggered first-treatment timing.")
        elif event_time_column:
            data_structure = "panel" if unit_id_column else "cross_sectional"
            study_design = "difference_in_differences_event_study"
            rationale.append("Relative-time field selects a dynamic Event Study.")
        elif post_column:
            data_structure = "panel" if unit_id_column else "cross_sectional"
            study_design = "difference_in_differences_two_period"
            rationale.append("Binary post indicator selects two-period DiD.")
        else:
            blocking.append(
                "DiD requires post_column, event_time_column, or complete staggered "
                "timing fields."
            )
    elif design_type == "instrumental_variable":
        study_design = "instrumental_variable"
        treatment_type = "continuous"
        if estimand != "LATE":
            blocking.append("Instrumental-variable analysis supports LATE only.")
        if not instruments:
            blocking.append("Instrumental-variable analysis requires instruments.")
        rationale.append("Explicit instruments select heteroskedasticity-robust 2SLS.")
    elif design_type == "regression_discontinuity":
        study_design = "regression_discontinuity"
        if estimand != "LATE":
            blocking.append("Regression discontinuity supports cutoff-local LATE only.")
        if not running_variable or cutoff is None:
            blocking.append("RDD requires running_variable and cutoff.")
        rationale.append("An explicit assignment cutoff selects sharp local-linear RDD.")
    elif design_type == "synthetic_control":
        data_structure = "panel"
        study_design = "synthetic_control_single_treated_unit"
        if estimand != "ATT":
            blocking.append("Synthetic control supports ATT only.")
        if not unit_id_column or not time_column or intervention_time is None:
            blocking.append(
                "Synthetic control requires unit_id_column, time_column, and "
                "intervention_time."
            )
        rationale.append(
            "One treated unit with an untreated donor pool selects synthetic control."
        )
    elif design_type == "time_series":
        data_structure = "time_series"
        study_design = "time_series_local_projection"
        treatment_type = "continuous"
        if estimand != "IRF":
            blocking.append("Time-series local projections support IRF only.")
        if not time_column:
            blocking.append("Time-series analysis requires time_column.")
        rationale.append(
            "One ordered series with a continuous shock selects local projections "
            "with lagged controls and HAC inference."
        )
    else:
        blocking.append(f"Unsupported design_type: {design_type}")

    selected = []
    if not blocking:
        selected = select_estimators(
            estimand=estimand,
            treatment_type=treatment_type,  # type: ignore[arg-type]
            data_structure=data_structure,  # type: ignore[arg-type]
            study_design=study_design,
            require_heterogeneity=estimand == "CATE",
        )
        if not selected:
            blocking.append("No registered estimator matches the explicit contract.")
    return {
        "ready": not blocking,
        "design_type": design_type,
        "study_design": study_design or None,
        "data_structure": data_structure,
        "estimand": estimand,
        "treatment_type": treatment_type,
        "selected_estimators": [spec.key for spec in selected],
        "selected_estimator_names": [spec.display_name for spec in selected],
        "primary_estimator": next(
            (spec.display_name for spec in selected if spec.role == "primary"), None
        ),
        "robustness_estimators": [
            spec.display_name for spec in selected if spec.role != "primary"
        ],
        "rationale": rationale,
        "blocking_issues": blocking,
    }
