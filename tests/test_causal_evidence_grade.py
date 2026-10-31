from __future__ import annotations

from causalens.causal.evidence_grade import build_causal_evidence_grade


def _primary(*, risk: str = "lower", interval=None, benchmark=None):
    return {
        "confidence_interval": interval or [1.0, 2.0],
        "sensitivity_analysis": {
            "available": True,
            "unmeasured_confounding_risk": risk,
            "observed_confounder_benchmark": benchmark
            or {
                "available": True,
                "assessment": "stronger_than_observed_benchmark_required",
            },
        },
    }


def _permutation(*, signal: str = "stronger_than_permuted_labels"):
    return {
        "available": True,
        "calibration_passed": True,
        "signal_status": signal,
    }


def test_grade_requires_negative_control_for_stronger_support() -> None:
    result = build_causal_evidence_grade(
        primary_estimate=_primary(),
        comparison={
            "estimate_count": 2,
            "confidence_intervals_overlap": True,
        },
        negative_control_summary={
            "requested_outcomes": [],
            "all_passed": None,
        },
        permutation_placebo=_permutation(),
    )

    assert result["grade"] == "moderate_support"
    assert "negative_control_outcome" in result["missing_checks"]


def test_grade_reaches_stronger_support_when_all_checks_pass() -> None:
    result = build_causal_evidence_grade(
        primary_estimate=_primary(),
        comparison={
            "estimate_count": 2,
            "confidence_intervals_overlap": True,
        },
        negative_control_summary={
            "requested_outcomes": ["baseline_outcome"],
            "all_passed": True,
        },
        permutation_placebo=_permutation(),
    )

    assert result["grade"] == "stronger_support"
    assert result["red_flags"] == []


def test_grade_is_limited_by_failed_negative_control() -> None:
    result = build_causal_evidence_grade(
        primary_estimate=_primary(),
        comparison={
            "estimate_count": 2,
            "confidence_intervals_overlap": True,
        },
        negative_control_summary={
            "requested_outcomes": ["baseline_outcome"],
            "all_passed": False,
        },
        permutation_placebo=_permutation(),
    )

    assert result["grade"] == "limited_support"
    assert "negative_control_outcome_failed" in result["red_flags"]


def test_grade_is_exploratory_when_primary_interval_includes_zero() -> None:
    result = build_causal_evidence_grade(
        primary_estimate=_primary(interval=[-0.5, 1.0]),
        comparison={"estimate_count": 1},
        negative_control_summary={},
        permutation_placebo=_permutation(
            signal="not_distinguishable_from_permuted_labels"
        ),
    )

    assert result["grade"] == "exploratory_only"


def test_temporal_grade_uses_pretrend_instead_of_treatment_permutation() -> None:
    primary = _primary()
    primary.update(
        {
            "estimator": "EventStudy",
            "pretrend_test": {"passed": True},
        }
    )
    result = build_causal_evidence_grade(
        primary_estimate=primary,
        comparison={"estimate_count": 1},
        negative_control_summary={
            "requested_outcomes": ["baseline_outcome"],
            "all_passed": True,
        },
        permutation_placebo={
            "available": False,
            "reason": "Not applicable to DiD designs.",
        },
    )

    assert result["components"]["design_family"] == "temporal"
    assert "treatment_permutation_diagnostic" not in result["missing_checks"]
    assert "pretrend_test_passed" in result["supporting_signals"]


def test_failed_pretrend_caps_temporal_grade() -> None:
    primary = _primary()
    primary.update(
        {
            "estimator": "EventStudy",
            "pretrend_test": {"passed": False},
        }
    )
    result = build_causal_evidence_grade(
        primary_estimate=primary,
        comparison={"estimate_count": 1},
        negative_control_summary={},
        permutation_placebo={"available": False},
    )

    assert result["grade"] == "limited_support"
    assert "pretrend_test_failed" in result["red_flags"]


def test_iv_grade_uses_first_stage_instead_of_regression_sensitivity() -> None:
    primary = {
        "estimator": "TwoStageLeastSquares",
        "confidence_interval": [1.0, 2.0],
        "sensitivity_analysis": {
            "available": False,
            "unmeasured_confounding_risk": "not_applicable",
        },
        "first_stage": {
            "f_statistic": 30.0,
            "partial_r2": 0.25,
            "weak_instrument_warning": False,
        },
        "overidentification_test": {
            "available": True,
            "passed": True,
            "p_value": 0.4,
        },
        "endogeneity_test": {"rejects_exogeneity_at_0_05": True},
    }
    result = build_causal_evidence_grade(
        primary_estimate=primary,
        comparison={"estimate_count": 1},
        negative_control_summary={},
        permutation_placebo={"available": False},
    )

    assert result["grade"] == "moderate_support"
    assert result["components"]["design_family"] == "instrumental_variable"
    assert "unmeasured_confounding_sensitivity" not in result["missing_checks"]
    assert "iv_first_stage_above_threshold" in result["supporting_signals"]


def test_weak_iv_first_stage_caps_grade() -> None:
    primary = {
        "estimator": "TwoStageLeastSquares",
        "confidence_interval": [1.0, 2.0],
        "first_stage": {"weak_instrument_warning": True},
    }
    result = build_causal_evidence_grade(
        primary_estimate=primary,
        comparison={"estimate_count": 1},
        negative_control_summary={},
        permutation_placebo={"available": False},
    )

    assert result["grade"] == "limited_support"
    assert "weak_instrument_first_stage" in result["red_flags"]


def test_rdd_grade_uses_design_specific_diagnostics() -> None:
    primary = {
        "estimator": "SharpRegressionDiscontinuity",
        "ate": 2.0,
        "confidence_interval": [1.5, 2.5],
        "density_diagnostic": {
            "available": True,
            "passed": True,
            "p_value": 0.5,
        },
        "covariate_continuity": {
            "available": True,
            "all_passed": True,
        },
        "bandwidth_sensitivity": {
            "available_count": 5,
            "direction_stable": True,
            "estimate_min": 1.8,
            "estimate_max": 2.2,
        },
        "donut_sensitivity": {"available": True, "estimate": 1.9},
    }
    result = build_causal_evidence_grade(
        primary_estimate=primary,
        comparison={"estimate_count": 1},
        negative_control_summary={},
        permutation_placebo={"available": False},
    )

    assert result["grade"] == "moderate_support"
    assert result["components"]["design_family"] == (
        "regression_discontinuity"
    )
    assert "rdd_density_check_passed" in result["supporting_signals"]
    assert "treatment_permutation_diagnostic" not in result["missing_checks"]


def test_rdd_density_failure_caps_grade() -> None:
    primary = {
        "estimator": "SharpRegressionDiscontinuity",
        "ate": 2.0,
        "confidence_interval": [1.5, 2.5],
        "density_diagnostic": {"available": True, "passed": False},
        "covariate_continuity": {"available": False},
        "bandwidth_sensitivity": {"direction_stable": True},
        "donut_sensitivity": {"available": True, "estimate": 2.1},
    }
    result = build_causal_evidence_grade(
        primary_estimate=primary,
        comparison={"estimate_count": 1},
        negative_control_summary={},
        permutation_placebo={"available": False},
    )

    assert result["grade"] == "limited_support"
    assert "rdd_density_discontinuity_detected" in result["red_flags"]


def test_synthetic_control_grade_uses_placebo_rank() -> None:
    primary = {
        "estimator": "SyntheticControl",
        "confidence_interval": [1.5, 3.0],
        "fit_diagnostics": {
            "pre_rmspe": 0.03,
            "post_pre_rmspe_ratio": 60.0,
        },
        "placebo_inference": {"placebo_count": 10, "p_value": 0.05},
    }
    result = build_causal_evidence_grade(
        primary_estimate=primary,
        comparison={"estimate_count": 1},
        negative_control_summary={},
        permutation_placebo={"available": False},
    )

    assert result["grade"] == "moderate_support"
    assert result["components"]["design_family"] == "synthetic_control"
    assert "synthetic_control_placebo_rank_supports_signal" in result[
        "supporting_signals"
    ]
    assert "treatment_permutation_diagnostic" not in result["missing_checks"]


def test_time_series_grade_uses_irf_diagnostics_and_is_capped() -> None:
    primary = {
        "estimator": "TimeSeriesLocalProjection",
        "confidence_interval": [0.2, 0.8],
        "stationarity": {
            "treatment": {
                "available": True,
                "p_value": 0.01,
                "reject_unit_root_at_10_percent": True,
            },
            "outcome": {
                "available": True,
                "p_value": 0.02,
                "reject_unit_root_at_10_percent": True,
            },
        },
        "placebo_test": {"passed": True, "p_value": 0.4},
        "residual_diagnostics": {
            "p_value": 0.3,
            "residual_autocorrelation_warning": False,
        },
    }
    result = build_causal_evidence_grade(
        primary_estimate=primary,
        comparison={"estimate_count": 1},
        negative_control_summary={},
        permutation_placebo={"available": False},
    )

    assert result["grade"] == "limited_support"
    assert result["components"]["design_family"] == "temporal"
    assert "pretrend_diagnostic" not in result["missing_checks"]
    assert "external_time_series_identification_argument" in result[
        "missing_checks"
    ]


def test_time_series_grade_is_exploratory_when_residuals_remain_correlated() -> None:
    primary = {
        "estimator": "TimeSeriesLocalProjection",
        "confidence_interval": [0.2, 0.8],
        "stationarity": {
            role: {
                "available": True,
                "p_value": 0.01,
                "reject_unit_root_at_10_percent": True,
            }
            for role in ("treatment", "outcome")
        },
        "placebo_test": {"passed": True},
        "residual_diagnostics": {
            "residual_autocorrelation_warning": True,
        },
    }
    result = build_causal_evidence_grade(
        primary_estimate=primary,
        comparison={"estimate_count": 1},
        negative_control_summary={},
        permutation_placebo={"available": False},
    )

    assert result["grade"] == "exploratory_only"
    assert "time_series_residual_autocorrelation_warning" in result["red_flags"]
