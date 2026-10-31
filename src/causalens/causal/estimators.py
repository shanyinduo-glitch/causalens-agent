from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal


TreatmentType = Literal["binary", "continuous"]
DataStructure = Literal["cross_sectional", "panel", "time_series"]


@dataclass(frozen=True)
class EstimatorSpec:
    key: str
    display_name: str
    supported_estimands: tuple[str, ...]
    treatment_types: tuple[TreatmentType, ...]
    data_structures: tuple[DataStructure, ...]
    study_designs: tuple[str, ...]
    supports_heterogeneity: bool
    role: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


ESTIMATOR_REGISTRY: dict[str, EstimatorSpec] = {
    "aipw": EstimatorSpec(
        key="aipw",
        display_name="AIPW",
        supported_estimands=("ATE",),
        treatment_types=("binary",),
        data_structures=("cross_sectional",),
        study_designs=("cross_sectional_adjustment",),
        supports_heterogeneity=False,
        role="primary",
    ),
    "linear_dml": EstimatorSpec(
        key="linear_dml",
        display_name="LinearDML",
        supported_estimands=("ATE",),
        treatment_types=("binary",),
        data_structures=("cross_sectional",),
        study_designs=("cross_sectional_adjustment",),
        supports_heterogeneity=False,
        role="robustness",
    ),
    "causal_forest_dml": EstimatorSpec(
        key="causal_forest_dml",
        display_name="CausalForestDML",
        supported_estimands=("CATE",),
        treatment_types=("binary",),
        data_structures=("cross_sectional",),
        study_designs=("cross_sectional_adjustment",),
        supports_heterogeneity=True,
        role="primary",
    ),
    "two_stage_least_squares": EstimatorSpec(
        key="two_stage_least_squares",
        display_name="TwoStageLeastSquares",
        supported_estimands=("LATE",),
        treatment_types=("binary", "continuous"),
        data_structures=("cross_sectional",),
        study_designs=("instrumental_variable",),
        supports_heterogeneity=False,
        role="primary",
    ),
    "sharp_regression_discontinuity": EstimatorSpec(
        key="sharp_regression_discontinuity",
        display_name="SharpRegressionDiscontinuity",
        supported_estimands=("LATE",),
        treatment_types=("binary",),
        data_structures=("cross_sectional",),
        study_designs=("regression_discontinuity",),
        supports_heterogeneity=False,
        role="primary",
    ),
    "synthetic_control": EstimatorSpec(
        key="synthetic_control",
        display_name="SyntheticControl",
        supported_estimands=("ATT",),
        treatment_types=("binary",),
        data_structures=("panel",),
        study_designs=("synthetic_control_single_treated_unit",),
        supports_heterogeneity=False,
        role="primary",
    ),
    "difference_in_differences": EstimatorSpec(
        key="difference_in_differences",
        display_name="DifferenceInDifferences",
        supported_estimands=("ATT",),
        treatment_types=("binary",),
        data_structures=("cross_sectional", "panel"),
        study_designs=("difference_in_differences_two_period",),
        supports_heterogeneity=False,
        role="primary",
    ),
    "event_study": EstimatorSpec(
        key="event_study",
        display_name="EventStudy",
        supported_estimands=("ATT",),
        treatment_types=("binary",),
        data_structures=("cross_sectional", "panel"),
        study_designs=("difference_in_differences_event_study",),
        supports_heterogeneity=False,
        role="primary",
    ),
    "staggered_adoption_did": EstimatorSpec(
        key="staggered_adoption_did",
        display_name="StaggeredAdoptionDID",
        supported_estimands=("ATT",),
        treatment_types=("binary",),
        data_structures=("panel",),
        study_designs=("difference_in_differences_staggered_adoption",),
        supports_heterogeneity=False,
        role="primary",
    ),
    "time_series_local_projection": EstimatorSpec(
        key="time_series_local_projection",
        display_name="TimeSeriesLocalProjection",
        supported_estimands=("IRF",),
        treatment_types=("continuous",),
        data_structures=("time_series",),
        study_designs=("time_series_local_projection",),
        supports_heterogeneity=False,
        role="primary",
    ),
}


def list_estimators() -> list[dict[str, object]]:
    """返回稳定、有序且可序列化的估计器能力清单。"""

    return [spec.to_dict() for spec in ESTIMATOR_REGISTRY.values()]


def select_estimators(
    *,
    estimand: str,
    treatment_type: TreatmentType,
    data_structure: DataStructure,
    study_design: str | None = None,
    require_heterogeneity: bool = False,
) -> list[EstimatorSpec]:
    """按明确的方法能力选择估计器，不把选择交给 LLM 猜测。"""

    selected = [
        spec
        for spec in ESTIMATOR_REGISTRY.values()
        if estimand in spec.supported_estimands
        and treatment_type in spec.treatment_types
        and data_structure in spec.data_structures
        and (study_design is None or study_design in spec.study_designs)
        and (
            not require_heterogeneity
            or spec.supports_heterogeneity
        )
    ]
    return sorted(
        selected,
        key=lambda spec: (spec.role != "primary", spec.key),
    )
