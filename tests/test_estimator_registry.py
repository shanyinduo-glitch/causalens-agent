from causalens.causal.estimators import (
    list_estimators,
    select_estimators,
)


def test_registry_selects_primary_then_robustness_estimator() -> None:
    selected = select_estimators(
        estimand="ATE",
        treatment_type="binary",
        data_structure="cross_sectional",
    )

    assert [spec.key for spec in selected] == ["aipw", "linear_dml"]
    assert selected[0].role == "primary"
    assert selected[1].role == "robustness"


def test_registry_exposes_serializable_capabilities() -> None:
    registry = list_estimators()

    assert [item["display_name"] for item in registry] == [
        "AIPW",
        "LinearDML",
        "CausalForestDML",
        "TwoStageLeastSquares",
        "SharpRegressionDiscontinuity",
        "SyntheticControl",
        "DifferenceInDifferences",
        "EventStudy",
        "StaggeredAdoptionDID",
        "TimeSeriesLocalProjection",
    ]
    assert all("supported_estimands" in item for item in registry)


def test_registry_selects_causal_forest_for_cate() -> None:
    selected = select_estimators(
        estimand="CATE",
        treatment_type="binary",
        data_structure="cross_sectional",
        require_heterogeneity=True,
    )

    assert [spec.key for spec in selected] == ["causal_forest_dml"]


def test_registry_selects_did_for_panel_att() -> None:
    selected = select_estimators(
        estimand="ATT",
        treatment_type="binary",
        data_structure="panel",
        study_design="difference_in_differences_two_period",
    )

    assert [spec.key for spec in selected] == ["difference_in_differences"]


def test_registry_selects_event_study_for_multi_period_att() -> None:
    selected = select_estimators(
        estimand="ATT",
        treatment_type="binary",
        data_structure="panel",
        study_design="difference_in_differences_event_study",
    )

    assert [spec.key for spec in selected] == ["event_study"]


def test_registry_selects_staggered_adoption_did() -> None:
    selected = select_estimators(
        estimand="ATT",
        treatment_type="binary",
        data_structure="panel",
        study_design="difference_in_differences_staggered_adoption",
    )

    assert [spec.key for spec in selected] == ["staggered_adoption_did"]


def test_registry_selects_two_stage_least_squares_for_iv_late() -> None:
    selected = select_estimators(
        estimand="LATE",
        treatment_type="continuous",
        data_structure="cross_sectional",
        study_design="instrumental_variable",
    )

    assert [spec.key for spec in selected] == ["two_stage_least_squares"]


def test_registry_selects_sharp_rdd_for_local_effect() -> None:
    selected = select_estimators(
        estimand="LATE",
        treatment_type="binary",
        data_structure="cross_sectional",
        study_design="regression_discontinuity",
    )

    assert [spec.key for spec in selected] == [
        "sharp_regression_discontinuity"
    ]


def test_registry_selects_synthetic_control_for_panel_att() -> None:
    selected = select_estimators(
        estimand="ATT",
        treatment_type="binary",
        data_structure="panel",
        study_design="synthetic_control_single_treated_unit",
    )

    assert [spec.key for spec in selected] == ["synthetic_control"]
