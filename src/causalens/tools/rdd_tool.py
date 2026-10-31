from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from pydantic import ValidationError
from scipy.stats import binomtest

from causalens.schemas import RDDReadinessResult, RDDResult


def _success(result: RDDReadinessResult | RDDResult) -> dict[str, Any]:
    return {"success": True, "result": result.model_dump(), "error": None}


def _error(message: str) -> dict[str, Any]:
    return {"success": False, "result": None, "error": message}


def _load_rdd_frame(
    *,
    dataset_path: str,
    treatment: str,
    outcome: str,
    running_variable: str,
    covariates: list[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if len(covariates) != len(set(covariates)):
        raise ValueError("covariates contains duplicates.")
    reserved = {treatment, outcome, running_variable}
    overlap = sorted(reserved & set(covariates))
    if overlap:
        raise ValueError(
            "RDD covariates must be distinct from treatment, outcome, and "
            f"running variable: {overlap}"
        )
    if len(reserved) != 3:
        raise ValueError(
            "RDD treatment, outcome, and running_variable must be distinct."
        )

    requested = [outcome, treatment, running_variable, *covariates]
    raw = pd.read_csv(dataset_path)
    missing = sorted(set(requested) - set(raw.columns))
    if missing:
        raise ValueError(f"RDD columns not found in CSV: {missing}")
    selected = raw[requested].replace([np.inf, -np.inf], np.nan)
    non_numeric = [
        column
        for column in requested
        if not pd.api.types.is_numeric_dtype(selected[column])
    ]
    if non_numeric:
        raise ValueError(
            "Sharp RDD currently requires numeric treatment, outcome, running "
            f"variable, and covariates: {non_numeric}"
        )
    complete = selected.dropna().astype(float)
    if complete.empty:
        raise ValueError("No complete observations are available for RDD analysis.")
    return complete, {
        "total_rows": int(len(raw)),
        "complete_cases": int(len(complete)),
        "dropped_rows": int(len(raw) - len(complete)),
        "complete_case_fraction": float(len(complete) / len(raw)),
    }


def _select_bandwidth(centered_running: np.ndarray) -> float:
    n = len(centered_running)
    scale = float(np.std(centered_running, ddof=1))
    if not np.isfinite(scale) or scale <= np.finfo(float).eps:
        raise ValueError("Running variable has insufficient variation.")
    bandwidth = 1.84 * scale * n ** (-0.2)
    positive_distances = np.abs(centered_running[np.abs(centered_running) > 0])
    if positive_distances.size == 0:
        raise ValueError("Running variable has no observations away from the cutoff.")
    return float(max(bandwidth, np.quantile(positive_distances, 0.10)))


def _assigned_treatment(
    centered_running: np.ndarray,
    treatment_above_cutoff: bool,
) -> np.ndarray:
    if treatment_above_cutoff:
        return (centered_running >= 0).astype(int)
    return (centered_running < 0).astype(int)


def _fit_local_linear(
    *,
    frame: pd.DataFrame,
    outcome: str,
    running_variable: str,
    cutoff: float,
    bandwidth: float,
    treatment_above_cutoff: bool,
    covariates: list[str],
    donut_radius: float = 0.0,
) -> dict[str, Any]:
    centered = frame[running_variable].to_numpy(dtype=float) - cutoff
    mask = (np.abs(centered) <= bandwidth) & (
        np.abs(centered) >= donut_radius
    )
    local = frame.loc[mask]
    local_running = centered[mask]
    if len(local) == 0:
        raise ValueError("No observations fall inside the requested RDD window.")
    treated_side = _assigned_treatment(
        local_running,
        treatment_above_cutoff,
    ).astype(float)
    left_count = int(np.sum(local_running < 0))
    right_count = int(np.sum(local_running >= 0))
    if min(left_count, right_count) < 5:
        raise ValueError("Too few observations on at least one side of the cutoff.")

    design_parts = [
        np.ones(len(local)),
        treated_side,
        local_running,
        treated_side * local_running,
    ]
    if covariates:
        design_parts.append(local[covariates].to_numpy(dtype=float))
    design = np.column_stack(design_parts)
    weights = np.clip(1.0 - np.abs(local_running) / bandwidth, 1e-8, None)
    fitted = sm.WLS(
        local[outcome].to_numpy(dtype=float),
        design,
        weights=weights,
    ).fit(cov_type="HC1")
    estimate = float(fitted.params[1])
    standard_error = float(fitted.bse[1])
    interval = [float(value) for value in fitted.conf_int(alpha=0.05)[1]]
    return {
        "estimate": estimate,
        "standard_error": standard_error,
        "confidence_interval": interval,
        "p_value": float(fitted.pvalues[1]),
        "sample_size": int(len(local)),
        "left_count": left_count,
        "right_count": right_count,
        "bandwidth": float(bandwidth),
        "donut_radius": float(donut_radius),
    }


def _density_count_diagnostic(
    centered_running: np.ndarray,
    bandwidth: float,
) -> dict[str, Any]:
    local = centered_running[np.abs(centered_running) <= bandwidth]
    left_count = int(np.sum(local < 0))
    right_count = int(np.sum(local >= 0))
    total = left_count + right_count
    if total == 0:
        return {"available": False, "reason": "No observations near cutoff."}
    test = binomtest(left_count, total, p=0.5, alternative="two-sided")
    return {
        "available": True,
        "method": "local_symmetric_count_test",
        "left_count": left_count,
        "right_count": right_count,
        "p_value": float(test.pvalue),
        "passed": bool(test.pvalue >= 0.05),
        "interpretation": (
            "This is a heuristic local count-symmetry diagnostic, not a formal "
            "McCrary density test and not proof that sorting is absent."
        ),
    }


def _covariate_continuity_diagnostics(
    *,
    frame: pd.DataFrame,
    running_variable: str,
    cutoff: float,
    bandwidth: float,
    treatment_above_cutoff: bool,
    covariates: list[str],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for covariate in covariates:
        result = _fit_local_linear(
            frame=frame,
            outcome=covariate,
            running_variable=running_variable,
            cutoff=cutoff,
            bandwidth=bandwidth,
            treatment_above_cutoff=treatment_above_cutoff,
            covariates=[],
        )
        rows.append(
            {
                "covariate": covariate,
                "discontinuity": result["estimate"],
                "standard_error": result["standard_error"],
                "confidence_interval": result["confidence_interval"],
                "p_value": result["p_value"],
                "passed": bool(result["p_value"] >= 0.05),
            }
        )
    return {
        "available": bool(covariates),
        "all_passed": all(row["passed"] for row in rows) if rows else None,
        "results": rows,
        "multiplicity_note": (
            "Covariates are checked individually without multiple-testing adjustment."
        ),
    }


def check_rdd_readiness(
    *,
    dataset_path: str,
    treatment: str,
    outcome: str,
    running_variable: str,
    cutoff: float,
    treatment_above_cutoff: bool = True,
    covariates: list[str] | None = None,
    bandwidth: float | None = None,
    min_observations_per_side: int = 30,
    minimum_assignment_compliance: float = 0.99,
) -> dict[str, Any]:
    """Validate a sharp RDD contract and local support around the cutoff."""

    try:
        if bandwidth is not None and bandwidth <= 0:
            raise ValueError("bandwidth must be positive when provided.")
        if min_observations_per_side < 5:
            raise ValueError("min_observations_per_side must be at least 5.")
        if not 0 < minimum_assignment_compliance <= 1:
            raise ValueError("minimum_assignment_compliance must be in (0, 1].")
        normalized_covariates = list(covariates or [])
        frame, sample_summary = _load_rdd_frame(
            dataset_path=dataset_path,
            treatment=treatment,
            outcome=outcome,
            running_variable=running_variable,
            covariates=normalized_covariates,
        )
        treatment_values = sorted(frame[treatment].unique().tolist())
        blocking_issues: list[str] = []
        warnings: list[str] = []
        if treatment_values != [0.0, 1.0]:
            blocking_issues.append(
                "Sharp RDD requires treatment encoded as binary 0/1."
            )

        centered = frame[running_variable].to_numpy(dtype=float) - float(cutoff)
        selected_bandwidth = float(
            bandwidth if bandwidth is not None else _select_bandwidth(centered)
        )
        local_mask = np.abs(centered) <= selected_bandwidth
        left_count = int(np.sum(local_mask & (centered < 0)))
        right_count = int(np.sum(local_mask & (centered >= 0)))
        if left_count < min_observations_per_side:
            blocking_issues.append(
                "Too few observations below the cutoff within the bandwidth."
            )
        if right_count < min_observations_per_side:
            blocking_issues.append(
                "Too few observations at or above the cutoff within the bandwidth."
            )

        assigned = _assigned_treatment(centered, treatment_above_cutoff)
        observed = frame[treatment].to_numpy(dtype=float)
        compliance = float(np.mean(observed == assigned))
        if compliance < minimum_assignment_compliance:
            blocking_issues.append(
                "Treatment does not follow the declared cutoff rule closely enough "
                "for sharp RDD; fuzzy RDD is not yet supported."
            )

        density = _density_count_diagnostic(centered, selected_bandwidth)
        if density.get("available") and density.get("passed") is False:
            warnings.append(
                "Observation counts are discontinuous around the cutoff under the "
                "heuristic density diagnostic; sorting or support imbalance is possible."
            )
        covariate_continuity: dict[str, Any] = {}
        if not blocking_issues:
            covariate_continuity = _covariate_continuity_diagnostics(
                frame=frame,
                running_variable=running_variable,
                cutoff=float(cutoff),
                bandwidth=selected_bandwidth,
                treatment_above_cutoff=treatment_above_cutoff,
                covariates=normalized_covariates,
            )
            if covariate_continuity.get("all_passed") is False:
                warnings.append(
                    "At least one declared pre-treatment covariate is discontinuous "
                    "at the cutoff."
                )

        result = RDDReadinessResult(
            ready=not blocking_issues,
            treatment=treatment,
            outcome=outcome,
            running_variable=running_variable,
            cutoff=float(cutoff),
            treatment_above_cutoff=treatment_above_cutoff,
            covariates=normalized_covariates,
            bandwidth=selected_bandwidth,
            sample_summary={
                **sample_summary,
                "left_count_within_bandwidth": left_count,
                "right_count_within_bandwidth": right_count,
                "assignment_compliance": compliance,
            },
            density_diagnostic=density,
            covariate_continuity=covariate_continuity,
            blocking_issues=blocking_issues,
            warnings=warnings,
            thresholds={
                "minimum_observations_per_side": min_observations_per_side,
                "minimum_assignment_compliance": minimum_assignment_compliance,
            },
        )
        return _success(result)
    except (ValueError, KeyError, FileNotFoundError, ValidationError) as exc:
        return _error(str(exc))
    except Exception as exc:
        return _error(f"RDD readiness check failed: {exc}")


def run_sharp_regression_discontinuity(
    *,
    dataset_path: str,
    treatment: str,
    outcome: str,
    running_variable: str,
    cutoff: float,
    treatment_above_cutoff: bool = True,
    covariates: list[str] | None = None,
    bandwidth: float | None = None,
    min_observations_per_side: int = 30,
) -> dict[str, Any]:
    """Estimate a covariate-adjusted local-linear sharp RDD effect at cutoff."""

    try:
        normalized_covariates = list(covariates or [])
        readiness_response = check_rdd_readiness(
            dataset_path=dataset_path,
            treatment=treatment,
            outcome=outcome,
            running_variable=running_variable,
            cutoff=cutoff,
            treatment_above_cutoff=treatment_above_cutoff,
            covariates=normalized_covariates,
            bandwidth=bandwidth,
            min_observations_per_side=min_observations_per_side,
        )
        if readiness_response.get("success") is not True:
            return readiness_response
        readiness = readiness_response["result"]
        if readiness.get("ready") is not True:
            return _error(
                "RDD readiness failed: "
                + "; ".join(readiness.get("blocking_issues") or [])
            )

        frame, _ = _load_rdd_frame(
            dataset_path=dataset_path,
            treatment=treatment,
            outcome=outcome,
            running_variable=running_variable,
            covariates=normalized_covariates,
        )
        selected_bandwidth = float(readiness["bandwidth"])
        primary = _fit_local_linear(
            frame=frame,
            outcome=outcome,
            running_variable=running_variable,
            cutoff=float(cutoff),
            bandwidth=selected_bandwidth,
            treatment_above_cutoff=treatment_above_cutoff,
            covariates=normalized_covariates,
        )

        sensitivity_rows: list[dict[str, Any]] = []
        for multiplier in (0.5, 0.75, 1.0, 1.25, 1.5):
            candidate = selected_bandwidth * multiplier
            try:
                estimate = _fit_local_linear(
                    frame=frame,
                    outcome=outcome,
                    running_variable=running_variable,
                    cutoff=float(cutoff),
                    bandwidth=candidate,
                    treatment_above_cutoff=treatment_above_cutoff,
                    covariates=normalized_covariates,
                )
                sensitivity_rows.append({"multiplier": multiplier, **estimate})
            except ValueError as exc:
                sensitivity_rows.append(
                    {
                        "multiplier": multiplier,
                        "bandwidth": candidate,
                        "available": False,
                        "reason": str(exc),
                    }
                )
        available_sensitivity = [
            row for row in sensitivity_rows if "estimate" in row
        ]
        signs = {int(np.sign(float(row["estimate"]))) for row in available_sensitivity}
        bandwidth_sensitivity = {
            "estimates": sensitivity_rows,
            "available_count": len(available_sensitivity),
            "direction_stable": len(signs - {0}) <= 1 and 0 not in signs,
            "estimate_min": min(
                (float(row["estimate"]) for row in available_sensitivity),
                default=None,
            ),
            "estimate_max": max(
                (float(row["estimate"]) for row in available_sensitivity),
                default=None,
            ),
        }

        donut_radius = selected_bandwidth * 0.05
        try:
            donut = {
                "available": True,
                **_fit_local_linear(
                    frame=frame,
                    outcome=outcome,
                    running_variable=running_variable,
                    cutoff=float(cutoff),
                    bandwidth=selected_bandwidth,
                    treatment_above_cutoff=treatment_above_cutoff,
                    covariates=normalized_covariates,
                    donut_radius=donut_radius,
                ),
            }
        except ValueError as exc:
            donut = {
                "available": False,
                "donut_radius": donut_radius,
                "reason": str(exc),
            }

        warnings = list(readiness.get("warnings") or [])
        if bandwidth_sensitivity["direction_stable"] is not True:
            warnings.append(
                "RDD effect direction is not stable across the configured "
                "bandwidth sensitivity grid."
            )
        warnings.append(
            "RDD causal interpretation requires potential outcomes to be smooth "
            "through the cutoff absent treatment and no precise manipulation of "
            "the running variable."
        )

        result = RDDResult(
            treatment=treatment,
            outcome=outcome,
            running_variable=running_variable,
            cutoff=float(cutoff),
            treatment_above_cutoff=treatment_above_cutoff,
            covariates=normalized_covariates,
            bandwidth=selected_bandwidth,
            sample_size=primary["sample_size"],
            ate=primary["estimate"],
            standard_error=primary["standard_error"],
            confidence_interval=primary["confidence_interval"],
            p_value=primary["p_value"],
            local_support={
                "left_count": primary["left_count"],
                "right_count": primary["right_count"],
            },
            density_diagnostic=readiness.get("density_diagnostic") or {},
            covariate_continuity=readiness.get("covariate_continuity") or {},
            bandwidth_sensitivity=bandwidth_sensitivity,
            donut_sensitivity=donut,
            model_configuration={
                "estimator": "local_linear_sharp_rdd",
                "kernel": "triangular",
                "bandwidth_selection": (
                    "user_supplied" if bandwidth is not None else "rule_of_thumb"
                ),
                "covariance": "HC1",
            },
            readiness_summary=readiness,
            assumptions=[
                "Treatment follows the declared sharp cutoff assignment rule.",
                "Potential outcomes are smooth through the cutoff absent treatment.",
                "Units cannot precisely manipulate the running variable around cutoff.",
                "No other intervention changes discontinuously at the same cutoff.",
                "The local observations near cutoff are measured consistently.",
            ],
            warnings=list(dict.fromkeys(warnings)),
        )
        return _success(result)
    except (ValueError, KeyError, FileNotFoundError, ValidationError) as exc:
        return _error(str(exc))
    except Exception as exc:
        return _error(f"Sharp regression discontinuity failed: {exc}")
