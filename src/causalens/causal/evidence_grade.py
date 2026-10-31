from __future__ import annotations

from collections.abc import Mapping
from math import isfinite
from typing import Any


GRADE_LABELS = {
    "exploratory_only": "可解释探索",
    "limited_support": "有限支持",
    "moderate_support": "中等支持",
    "stronger_support": "较强支持",
    "blocked": "被诊断阻断",
}


def _interval_excludes_zero(interval: object) -> bool | None:
    if not isinstance(interval, (list, tuple)) or len(interval) != 2:
        return None
    try:
        lower, upper = float(interval[0]), float(interval[1])
    except (TypeError, ValueError):
        return None
    if not isfinite(lower) or not isfinite(upper) or lower > upper:
        return None
    return lower > 0 or upper < 0


def build_causal_evidence_grade(
    *,
    primary_estimate: Mapping[str, Any],
    comparison: Mapping[str, Any],
    negative_control_summary: Mapping[str, Any],
    permutation_placebo: Mapping[str, Any],
) -> dict[str, Any]:
    """把估计、稳健性和 falsification 诊断汇总为可审计等级。"""

    red_flags: list[str] = []
    supporting_signals: list[str] = []
    missing_checks: list[str] = []
    components: dict[str, Any] = {}

    estimator = str(primary_estimate.get("estimator") or "")
    temporal_estimators = {
        "DifferenceInDifferences",
        "EventStudy",
        "StaggeredAdoptionDID",
        "TimeSeriesLocalProjection",
    }
    is_iv_design = estimator == "TwoStageLeastSquares"
    is_rdd_design = estimator == "SharpRegressionDiscontinuity"
    is_synthetic_control = estimator == "SyntheticControl"
    is_time_series = estimator == "TimeSeriesLocalProjection"
    pretrend = primary_estimate.get("pretrend_test") or {}
    placebo = primary_estimate.get("placebo_test") or {}
    is_temporal_design = (
        estimator in temporal_estimators or bool(pretrend) or bool(placebo)
    )
    components["design_family"] = (
        "instrumental_variable"
        if is_iv_design
        else "regression_discontinuity"
        if is_rdd_design
        else "synthetic_control"
        if is_synthetic_control
        else "temporal"
        if is_temporal_design
        else "cross_sectional"
    )

    excludes_zero = _interval_excludes_zero(
        primary_estimate.get("confidence_interval")
    )
    components["primary_interval_excludes_zero"] = excludes_zero
    if excludes_zero is True:
        supporting_signals.append("primary_interval_excludes_zero")
    elif excludes_zero is False:
        red_flags.append("primary_interval_includes_zero")
    else:
        missing_checks.append("primary_confidence_interval")

    estimate_count = int(comparison.get("estimate_count") or 1)
    intervals_overlap = comparison.get("confidence_intervals_overlap")
    components["estimator_agreement"] = {
        "estimate_count": estimate_count,
        "confidence_intervals_overlap": intervals_overlap,
    }
    if estimate_count >= 2:
        if intervals_overlap is True:
            supporting_signals.append("robustness_estimator_intervals_overlap")
        elif intervals_overlap is False:
            red_flags.append("robustness_estimator_intervals_do_not_overlap")
    else:
        missing_checks.append("robustness_estimator_comparison")

    sensitivity = primary_estimate.get("sensitivity_analysis") or {}
    if is_iv_design or is_rdd_design or is_synthetic_control or is_time_series:
        sensitivity_risk = "not_applicable"
        components["unmeasured_confounding_sensitivity_not_applicable"] = True
    elif not isinstance(sensitivity, Mapping) or not sensitivity.get("available"):
        missing_checks.append("unmeasured_confounding_sensitivity")
        sensitivity_risk = "unavailable"
    else:
        sensitivity_risk = str(
            sensitivity.get("unmeasured_confounding_risk") or "unavailable"
        )
        if sensitivity_risk in {"lower", "moderate"}:
            supporting_signals.append(
                f"unmeasured_confounding_risk_{sensitivity_risk}"
            )
        elif sensitivity_risk in {
            "high",
            "elevated",
            "statistically_inconclusive",
        }:
            red_flags.append(
                f"unmeasured_confounding_risk_{sensitivity_risk}"
            )
        else:
            missing_checks.append("unmeasured_confounding_risk_classification")
    components["unmeasured_confounding_risk"] = sensitivity_risk

    benchmark = (
        sensitivity.get("observed_confounder_benchmark") or {}
        if isinstance(sensitivity, Mapping)
        else {}
    )
    benchmark_assessment = benchmark.get("assessment")
    components["observed_confounder_benchmark"] = benchmark_assessment
    if is_iv_design or is_rdd_design or is_synthetic_control or is_time_series:
        components["observed_confounder_benchmark_not_applicable"] = True
    elif benchmark.get("available"):
        if benchmark_assessment == "stronger_than_observed_benchmark_required":
            supporting_signals.append("stronger_than_observed_benchmark_required")
        elif benchmark_assessment in {
            "observed_strength_can_explain_away_point_estimate",
            "observed_strength_can_remove_significance",
        }:
            red_flags.append(str(benchmark_assessment))
    else:
        missing_checks.append("observed_confounder_benchmark")

    permutation_available = permutation_placebo.get("available") is True
    permutation_status = permutation_placebo.get("signal_status")
    calibration_passed = permutation_placebo.get("calibration_passed")
    components["permutation_placebo"] = {
        "available": permutation_available,
        "signal_status": permutation_status,
        "calibration_passed": calibration_passed,
    }
    if permutation_available:
        if calibration_passed is not True:
            red_flags.append("permutation_distribution_not_centered")
        elif permutation_status == "stronger_than_permuted_labels":
            supporting_signals.append("stronger_than_permuted_labels")
        else:
            red_flags.append("not_distinguishable_from_permuted_labels")
    elif (
        not is_temporal_design
        and not is_iv_design
        and not is_rdd_design
        and not is_synthetic_control
    ):
        missing_checks.append("treatment_permutation_diagnostic")
    else:
        components["permutation_placebo"]["not_applicable"] = True

    if is_iv_design:
        first_stage = primary_estimate.get("first_stage") or {}
        weak_instrument = first_stage.get("weak_instrument_warning")
        components["iv_first_stage"] = {
            "f_statistic": first_stage.get("f_statistic"),
            "partial_r2": first_stage.get("partial_r2"),
            "weak_instrument_warning": weak_instrument,
        }
        if weak_instrument is False:
            supporting_signals.append("iv_first_stage_above_threshold")
        elif weak_instrument is True:
            red_flags.append("weak_instrument_first_stage")
        else:
            missing_checks.append("iv_first_stage_diagnostic")

        overidentification = (
            primary_estimate.get("overidentification_test") or {}
        )
        components["iv_overidentification_test"] = {
            "available": overidentification.get("available"),
            "passed": overidentification.get("passed"),
            "p_value": overidentification.get("p_value"),
        }
        if overidentification.get("available"):
            if overidentification.get("passed") is True:
                supporting_signals.append(
                    "overidentifying_restrictions_not_rejected"
                )
            else:
                red_flags.append("overidentifying_restrictions_rejected")

        endogeneity = primary_estimate.get("endogeneity_test") or {}
        components["iv_endogeneity_test"] = {
            "p_value": endogeneity.get("p_value"),
            "rejects_exogeneity_at_0_05": endogeneity.get(
                "rejects_exogeneity_at_0_05"
            ),
        }
        if endogeneity.get("rejects_exogeneity_at_0_05") is True:
            supporting_signals.append("treatment_endogeneity_detected")

    if is_rdd_design:
        density = primary_estimate.get("density_diagnostic") or {}
        components["rdd_density_diagnostic"] = {
            "available": density.get("available"),
            "passed": density.get("passed"),
            "p_value": density.get("p_value"),
            "method": density.get("method"),
        }
        if density.get("available"):
            if density.get("passed") is True:
                supporting_signals.append("rdd_density_check_passed")
            else:
                red_flags.append("rdd_density_discontinuity_detected")
        else:
            missing_checks.append("rdd_density_diagnostic")

        covariate_continuity = (
            primary_estimate.get("covariate_continuity") or {}
        )
        components["rdd_covariate_continuity"] = {
            "available": covariate_continuity.get("available"),
            "all_passed": covariate_continuity.get("all_passed"),
        }
        if covariate_continuity.get("available"):
            if covariate_continuity.get("all_passed") is True:
                supporting_signals.append("rdd_covariate_continuity_passed")
            else:
                red_flags.append("rdd_covariate_discontinuity_detected")
        else:
            missing_checks.append("rdd_covariate_continuity")

        bandwidth = primary_estimate.get("bandwidth_sensitivity") or {}
        components["rdd_bandwidth_sensitivity"] = {
            "available_count": bandwidth.get("available_count"),
            "direction_stable": bandwidth.get("direction_stable"),
            "estimate_min": bandwidth.get("estimate_min"),
            "estimate_max": bandwidth.get("estimate_max"),
        }
        if bandwidth.get("direction_stable") is True:
            supporting_signals.append("rdd_bandwidth_direction_stable")
        elif bandwidth.get("direction_stable") is False:
            red_flags.append("rdd_bandwidth_direction_unstable")
        else:
            missing_checks.append("rdd_bandwidth_sensitivity")

        donut = primary_estimate.get("donut_sensitivity") or {}
        donut_estimate = donut.get("estimate")
        primary_effect = primary_estimate.get("ate")
        if donut.get("available") and donut_estimate is not None:
            direction_stable = bool(
                float(donut_estimate) * float(primary_effect) > 0
            )
            components["rdd_donut_direction_stable"] = direction_stable
            if direction_stable:
                supporting_signals.append("rdd_donut_direction_stable")
            else:
                red_flags.append("rdd_donut_direction_unstable")
        else:
            missing_checks.append("rdd_donut_sensitivity")

    if is_synthetic_control:
        fit = primary_estimate.get("fit_diagnostics") or {}
        placebo_inference = primary_estimate.get("placebo_inference") or {}
        placebo_count = int(placebo_inference.get("placebo_count") or 0)
        placebo_p_value = placebo_inference.get("p_value")
        components["synthetic_control_fit"] = {
            "pre_rmspe": fit.get("pre_rmspe"),
            "post_rmspe": fit.get("post_rmspe"),
            "post_pre_rmspe_ratio": fit.get("post_pre_rmspe_ratio"),
        }
        components["synthetic_control_placebo_inference"] = {
            "placebo_count": placebo_count,
            "p_value": placebo_p_value,
        }
        if fit.get("pre_rmspe") is None:
            missing_checks.append("synthetic_control_pre_fit")
        else:
            supporting_signals.append("synthetic_control_pre_fit_reported")
        if placebo_count < 5 or placebo_p_value is None:
            missing_checks.append("synthetic_control_placebo_inference")
        elif float(placebo_p_value) <= 0.10:
            supporting_signals.append("synthetic_control_placebo_rank_supports_signal")
        else:
            red_flags.append("synthetic_control_placebo_rank_not_distinct")

    if is_time_series:
        stationarity = primary_estimate.get("stationarity") or {}
        stationarity_status: dict[str, Any] = {}
        for role in ("treatment", "outcome"):
            diagnostic = stationarity.get(role) or {}
            passed = diagnostic.get("reject_unit_root_at_10_percent")
            stationarity_status[role] = {
                "available": diagnostic.get("available"),
                "p_value": diagnostic.get("p_value"),
                "reject_unit_root_at_10_percent": passed,
            }
            if passed is True:
                supporting_signals.append(f"time_series_{role}_adf_warning_check_passed")
            elif passed is False:
                red_flags.append(f"time_series_{role}_unit_root_warning")
            else:
                missing_checks.append(f"time_series_{role}_stationarity_check")
        components["time_series_stationarity"] = stationarity_status

        residual = primary_estimate.get("residual_diagnostics") or {}
        autocorrelation_warning = residual.get(
            "residual_autocorrelation_warning"
        )
        components["time_series_residual_diagnostics"] = {
            "available": bool(residual),
            "p_value": residual.get("p_value"),
            "residual_autocorrelation_warning": autocorrelation_warning,
        }
        if autocorrelation_warning is False:
            supporting_signals.append("time_series_residual_autocorrelation_check_passed")
        elif autocorrelation_warning is True:
            red_flags.append("time_series_residual_autocorrelation_warning")
        else:
            missing_checks.append("time_series_residual_autocorrelation_check")
        missing_checks.append("external_time_series_identification_argument")

    requested_negative_controls = list(
        negative_control_summary.get("requested_outcomes") or []
    )
    negative_all_passed = negative_control_summary.get("all_passed")
    components["negative_controls"] = {
        "requested_count": len(requested_negative_controls),
        "all_passed": negative_all_passed,
    }
    if requested_negative_controls:
        if negative_all_passed is True:
            supporting_signals.append("all_declared_negative_controls_passed")
        else:
            red_flags.append("negative_control_outcome_failed")
    else:
        missing_checks.append("negative_control_outcome")

    if isinstance(pretrend, Mapping) and pretrend:
        components["pretrend_test_passed"] = pretrend.get("passed")
        if pretrend.get("passed") is True:
            supporting_signals.append("pretrend_test_passed")
        elif pretrend.get("passed") is False:
            red_flags.append("pretrend_test_failed")
        else:
            missing_checks.append("pretrend_diagnostic")
    elif is_temporal_design and not is_time_series:
        missing_checks.append("pretrend_diagnostic")
    if isinstance(placebo, Mapping) and placebo:
        components["timing_placebo_passed"] = placebo.get("passed")
        if placebo.get("passed") is True:
            supporting_signals.append("timing_placebo_passed")
        elif placebo.get("passed") is False:
            red_flags.append("timing_placebo_failed")
        elif estimator == "StaggeredAdoptionDID":
            missing_checks.append("timing_placebo_diagnostic")

    if is_time_series:
        stationarity_components = components.get("time_series_stationarity") or {}
        stationarity_passed = all(
            (stationarity_components.get(role) or {}).get(
                "reject_unit_root_at_10_percent"
            )
            is True
            for role in ("treatment", "outcome")
        )
        placebo_passed = placebo.get("passed") is True
        residual_ok = (
            components.get("time_series_residual_diagnostics") or {}
        ).get("residual_autocorrelation_warning") is False
        grade = (
            "limited_support"
            if excludes_zero is True
            and stationarity_passed
            and placebo_passed
            and residual_ok
            else "exploratory_only"
        )
    elif excludes_zero is not True or (
        not is_iv_design
        and not is_rdd_design
        and not is_synthetic_control
        and sensitivity_risk in {
        "statistically_inconclusive",
        "unavailable",
        }
    ):
        grade = "exploratory_only"
    elif red_flags:
        grade = "limited_support"
    else:
        strongest_requirements = {
            "robustness_estimator_intervals_overlap",
            "unmeasured_confounding_risk_lower",
            "stronger_than_observed_benchmark_required",
            "all_declared_negative_controls_passed",
        }
        if is_temporal_design:
            strongest_requirements.add("pretrend_test_passed")
            if estimator == "StaggeredAdoptionDID":
                strongest_requirements.add("timing_placebo_passed")
        else:
            strongest_requirements.add("stronger_than_permuted_labels")
        grade = (
            "stronger_support"
            if strongest_requirements.issubset(supporting_signals)
            else "moderate_support"
        )

    return {
        "grade": grade,
        "label_zh": GRADE_LABELS[grade],
        "ordinal_rank": {
            "blocked": 0,
            "exploratory_only": 1,
            "limited_support": 2,
            "moderate_support": 3,
            "stronger_support": 4,
        }[grade],
        "components": components,
        "supporting_signals": supporting_signals,
        "red_flags": red_flags,
        "missing_checks": missing_checks,
        "interpretation": (
            "This deterministic grade summarizes available diagnostics. It is "
            "not a probability that the causal claim is true and not proof of "
            "identification."
        ),
    }
