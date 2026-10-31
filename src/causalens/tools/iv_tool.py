from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from pydantic import ValidationError
from scipy.stats import chi2, norm

from causalens.schemas import IVReadinessResult, IVResult


def _success(result: IVReadinessResult | IVResult) -> dict[str, Any]:
    return {"success": True, "result": result.model_dump(), "error": None}


def _error(message: str) -> dict[str, Any]:
    return {"success": False, "result": None, "error": message}


def _load_iv_frame(
    *,
    dataset_path: str,
    treatment: str,
    outcome: str,
    instruments: list[str],
    confounders: list[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not instruments:
        raise ValueError("IV design requires at least one declared instrument.")
    if len(instruments) != len(set(instruments)):
        raise ValueError("instruments contains duplicates.")
    if len(confounders) != len(set(confounders)):
        raise ValueError("confounders contains duplicates.")

    forbidden = {treatment, outcome, *confounders} & set(instruments)
    if forbidden:
        raise ValueError(
            "Instruments must be distinct from treatment, outcome, and "
            f"confounders: {sorted(forbidden)}"
        )

    requested = list(dict.fromkeys([outcome, treatment, *instruments, *confounders]))
    raw = pd.read_csv(dataset_path)
    missing = sorted(set(requested) - set(raw.columns))
    if missing:
        raise ValueError(f"IV columns not found in CSV: {missing}")

    selected = raw[requested].replace([np.inf, -np.inf], np.nan)
    non_numeric = [
        column
        for column in requested
        if not pd.api.types.is_numeric_dtype(selected[column])
    ]
    if non_numeric:
        raise ValueError(
            "IV/2SLS currently requires numeric treatment, outcome, "
            f"instruments, and confounders: {non_numeric}"
        )

    complete = selected.dropna().astype(float)
    if complete.empty:
        raise ValueError("No complete observations are available for IV analysis.")

    summary = {
        "total_rows": int(len(raw)),
        "complete_cases": int(len(complete)),
        "dropped_rows": int(len(raw) - len(complete)),
        "complete_case_fraction": float(len(complete) / len(raw)),
    }
    return complete, summary


def _first_stage_diagnostics(
    *,
    frame: pd.DataFrame,
    treatment: str,
    instruments: list[str],
    confounders: list[str],
    weak_instrument_f_threshold: float,
) -> tuple[dict[str, Any], np.ndarray]:
    exogenous = sm.add_constant(
        frame[confounders],
        has_constant="add",
    ).to_numpy(dtype=float)
    excluded = frame[instruments].to_numpy(dtype=float)
    full_design = np.column_stack([exogenous, excluded])
    treatment_values = frame[treatment].to_numpy(dtype=float)

    first_stage = sm.OLS(treatment_values, full_design).fit(cov_type="HC1")
    restriction = np.zeros((len(instruments), full_design.shape[1]))
    restriction[:, exogenous.shape[1] :] = np.eye(len(instruments))
    joint_test = first_stage.f_test(restriction)
    f_statistic = float(np.asarray(joint_test.fvalue).reshape(-1)[0])
    f_p_value = float(np.asarray(joint_test.pvalue).reshape(-1)[0])

    reduced = sm.OLS(treatment_values, exogenous).fit()
    full_unadjusted = sm.OLS(treatment_values, full_design).fit()
    reduced_ssr = float(np.square(reduced.resid).sum())
    full_ssr = float(np.square(full_unadjusted.resid).sum())
    partial_r2 = (
        float(max(0.0, min(1.0, (reduced_ssr - full_ssr) / reduced_ssr)))
        if reduced_ssr > np.finfo(float).eps
        else 0.0
    )

    return (
        {
            "excluded_instrument_count": len(instruments),
            "f_statistic": f_statistic,
            "f_p_value": f_p_value,
            "partial_r2": partial_r2,
            "threshold": float(weak_instrument_f_threshold),
            "weak_instrument_warning": bool(
                not np.isfinite(f_statistic)
                or f_statistic < weak_instrument_f_threshold
            ),
            "instrument_coefficients": {
                instrument: float(first_stage.params[exogenous.shape[1] + index])
                for index, instrument in enumerate(instruments)
            },
            "first_stage_r_squared": float(first_stage.rsquared),
        },
        np.asarray(full_unadjusted.resid, dtype=float),
    )


def check_iv_readiness(
    *,
    dataset_path: str,
    treatment: str,
    outcome: str,
    instruments: list[str],
    confounders: list[str] | None = None,
    min_complete_cases: int = 50,
    weak_instrument_f_threshold: float = 10.0,
) -> dict[str, Any]:
    """Check the order/rank conditions and empirical first-stage relevance."""

    try:
        if min_complete_cases < 10:
            raise ValueError("min_complete_cases must be at least 10.")
        if weak_instrument_f_threshold <= 0:
            raise ValueError("weak_instrument_f_threshold must be positive.")

        normalized_confounders = list(confounders or [])
        frame, sample_summary = _load_iv_frame(
            dataset_path=dataset_path,
            treatment=treatment,
            outcome=outcome,
            instruments=list(instruments),
            confounders=normalized_confounders,
        )

        blocking_issues: list[str] = []
        warnings: list[str] = []
        if len(frame) < min_complete_cases:
            blocking_issues.append(
                "Complete-case sample size is below the configured IV minimum."
            )
        if sample_summary["dropped_rows"]:
            warnings.append(
                "Rows with missing or non-finite IV analysis values were dropped."
            )
        if frame[treatment].nunique() < 2:
            blocking_issues.append("Treatment has no usable variation.")

        constant_instruments = [
            instrument
            for instrument in instruments
            if frame[instrument].nunique() < 2
        ]
        if constant_instruments:
            blocking_issues.append(
                f"Instruments have no usable variation: {constant_instruments}"
            )

        exogenous = sm.add_constant(
            frame[normalized_confounders],
            has_constant="add",
        ).to_numpy(dtype=float)
        instrument_matrix = np.column_stack(
            [exogenous, frame[instruments].to_numpy(dtype=float)]
        )
        matrix_rank = int(np.linalg.matrix_rank(instrument_matrix))
        if matrix_rank < instrument_matrix.shape[1]:
            blocking_issues.append(
                "The instrument matrix is rank deficient after conditioning on "
                "the declared confounders."
            )

        first_stage: dict[str, Any] = {}
        if not blocking_issues:
            first_stage, _ = _first_stage_diagnostics(
                frame=frame,
                treatment=treatment,
                instruments=list(instruments),
                confounders=normalized_confounders,
                weak_instrument_f_threshold=weak_instrument_f_threshold,
            )
            if first_stage["weak_instrument_warning"]:
                warnings.append(
                    "The excluded-instrument first-stage F statistic is below "
                    "the configured rule-of-thumb threshold; 2SLS may be weak-"
                    "instrument biased."
                )

        result = IVReadinessResult(
            ready=not blocking_issues,
            treatment=treatment,
            outcome=outcome,
            instruments=list(instruments),
            confounders=normalized_confounders,
            sample_summary={
                **sample_summary,
                "instrument_matrix_rank": matrix_rank,
                "instrument_matrix_columns": int(instrument_matrix.shape[1]),
            },
            first_stage=first_stage,
            blocking_issues=blocking_issues,
            warnings=warnings,
            thresholds={
                "minimum_complete_cases": min_complete_cases,
                "weak_instrument_f_statistic": weak_instrument_f_threshold,
            },
        )
        return _success(result)
    except (ValueError, KeyError, FileNotFoundError, ValidationError) as exc:
        return _error(str(exc))
    except Exception as exc:
        return _error(f"IV readiness check failed: {exc}")


def _heteroskedastic_2sls(
    *,
    outcome: np.ndarray,
    structural: np.ndarray,
    instruments: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(outcome)
    ztz_inverse = np.linalg.pinv(instruments.T @ instruments)
    cross = structural.T @ instruments
    normal = cross @ ztz_inverse @ cross.T
    beta = np.linalg.pinv(normal) @ (
        cross @ ztz_inverse @ instruments.T @ outcome
    )
    residuals = outcome - structural @ beta

    a_matrix = structural.T @ instruments / n
    b_inverse = np.linalg.pinv(instruments.T @ instruments / n)
    omega = (
        instruments.T
        @ (np.square(residuals)[:, None] * instruments)
        / n
    )
    bread = np.linalg.pinv(a_matrix @ b_inverse @ a_matrix.T)
    covariance = (
        bread
        @ a_matrix
        @ b_inverse
        @ omega
        @ b_inverse
        @ a_matrix.T
        @ bread
        / n
    )
    covariance *= n / max(n - structural.shape[1], 1)
    return beta, covariance, residuals


def run_two_stage_least_squares(
    *,
    dataset_path: str,
    treatment: str,
    outcome: str,
    instruments: list[str],
    confounders: list[str] | None = None,
    confidence_level: float = 0.95,
    min_complete_cases: int = 50,
    weak_instrument_f_threshold: float = 10.0,
) -> dict[str, Any]:
    """Estimate a single-endogenous-treatment IV effect with robust 2SLS SEs."""

    try:
        if not 0 < confidence_level < 1:
            raise ValueError("confidence_level must be between 0 and 1.")
        normalized_confounders = list(confounders or [])
        readiness_response = check_iv_readiness(
            dataset_path=dataset_path,
            treatment=treatment,
            outcome=outcome,
            instruments=list(instruments),
            confounders=normalized_confounders,
            min_complete_cases=min_complete_cases,
            weak_instrument_f_threshold=weak_instrument_f_threshold,
        )
        if readiness_response.get("success") is not True:
            return readiness_response
        readiness = readiness_response["result"]
        if readiness.get("ready") is not True:
            return _error(
                "IV readiness failed: "
                + "; ".join(readiness.get("blocking_issues") or [])
            )

        frame, _ = _load_iv_frame(
            dataset_path=dataset_path,
            treatment=treatment,
            outcome=outcome,
            instruments=list(instruments),
            confounders=normalized_confounders,
        )
        exogenous = sm.add_constant(
            frame[normalized_confounders],
            has_constant="add",
        ).to_numpy(dtype=float)
        treatment_values = frame[treatment].to_numpy(dtype=float)
        outcome_values = frame[outcome].to_numpy(dtype=float)
        excluded = frame[instruments].to_numpy(dtype=float)
        structural = np.column_stack([exogenous, treatment_values])
        instrument_matrix = np.column_stack([exogenous, excluded])

        beta, covariance, residuals = _heteroskedastic_2sls(
            outcome=outcome_values,
            structural=structural,
            instruments=instrument_matrix,
        )
        treatment_index = structural.shape[1] - 1
        effect = float(beta[treatment_index])
        variance = float(covariance[treatment_index, treatment_index])
        standard_error = float(np.sqrt(max(variance, 0.0)))
        critical_value = float(norm.ppf(0.5 + confidence_level / 2.0))
        interval = [
            effect - critical_value * standard_error,
            effect + critical_value * standard_error,
        ]

        first_stage, first_stage_residuals = _first_stage_diagnostics(
            frame=frame,
            treatment=treatment,
            instruments=list(instruments),
            confounders=normalized_confounders,
            weak_instrument_f_threshold=weak_instrument_f_threshold,
        )

        control_function_design = np.column_stack(
            [structural, first_stage_residuals]
        )
        endogeneity_fit = sm.OLS(
            outcome_values,
            control_function_design,
        ).fit(cov_type="HC1")
        endogeneity_p_value = float(endogeneity_fit.pvalues[-1])
        endogeneity_test = {
            "method": "durbin_wu_hausman_control_function",
            "null_hypothesis": "Treatment is exogenous conditional on covariates.",
            "statistic": float(endogeneity_fit.tvalues[-1] ** 2),
            "p_value": endogeneity_p_value,
            "rejects_exogeneity_at_0_05": bool(endogeneity_p_value < 0.05),
        }

        overidentification: dict[str, Any]
        overidentification_df = len(instruments) - 1
        if overidentification_df > 0:
            moment_mean = instrument_matrix.T @ residuals / len(frame)
            omega = (
                instrument_matrix.T
                @ (np.square(residuals)[:, None] * instrument_matrix)
                / len(frame)
            )
            statistic = float(
                len(frame) * moment_mean.T @ np.linalg.pinv(omega) @ moment_mean
            )
            p_value = float(chi2.sf(statistic, overidentification_df))
            overidentification = {
                "available": True,
                "method": "heteroskedasticity_robust_hansen_j",
                "statistic": statistic,
                "degrees_of_freedom": overidentification_df,
                "p_value": p_value,
                "passed": bool(p_value >= 0.05),
                "interpretation": (
                    "Failure to reject does not prove the exclusion restriction "
                    "or instrument independence."
                ),
            }
        else:
            overidentification = {
                "available": False,
                "reason": "Exactly identified models have no overidentifying restrictions.",
            }

        warnings = list(readiness.get("warnings") or [])
        if overidentification.get("available") and not overidentification.get(
            "passed"
        ):
            warnings.append(
                "The Hansen J test rejects the joint overidentifying restrictions; "
                "at least one instrument or model restriction may be invalid."
            )
        warnings.append(
            "Instrument independence and the exclusion restriction cannot be "
            "established from the observed CSV alone."
        )

        result = IVResult(
            treatment=treatment,
            outcome=outcome,
            instruments=list(instruments),
            confounders=normalized_confounders,
            sample_size=int(len(frame)),
            ate=effect,
            standard_error=standard_error,
            confidence_interval=[float(value) for value in interval],
            confidence_level=confidence_level,
            first_stage=first_stage,
            endogeneity_test=endogeneity_test,
            overidentification_test=overidentification,
            model_configuration={
                "estimator": "heteroskedasticity_robust_2sls",
                "endogenous_treatment_count": 1,
                "excluded_instrument_count": len(instruments),
                "covariance": "HC1 sandwich",
            },
            readiness_summary=readiness,
            assumptions=[
                "Instrument relevance after conditioning on declared confounders.",
                "Instrument independence from unmeasured causes of the outcome.",
                "Exclusion restriction: instruments affect outcome only through treatment.",
                "Monotonicity for a LATE interpretation with binary encouragement.",
                "Stable treatment and no interference between analysis units.",
            ],
            warnings=list(dict.fromkeys(warnings)),
        )
        return _success(result)
    except (ValueError, KeyError, FileNotFoundError, ValidationError) as exc:
        return _error(str(exc))
    except Exception as exc:
        return _error(f"Two-stage least squares failed: {exc}")
