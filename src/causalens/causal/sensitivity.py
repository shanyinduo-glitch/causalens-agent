from __future__ import annotations

from math import sqrt
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import t as student_t


def robustness_value(
    *,
    t_statistic: float,
    degrees_of_freedom: int,
    alpha: float | None = None,
) -> float:
    """计算 Cinelli-Hazlett partial-R² robustness value。"""

    if degrees_of_freedom < 2:
        raise ValueError("degrees_of_freedom must be at least 2.")
    if alpha is not None and not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1.")

    partial_f = abs(float(t_statistic)) / sqrt(degrees_of_freedom)
    if alpha is not None:
        critical_f = float(
            student_t.ppf(1 - alpha / 2, degrees_of_freedom - 1)
        ) / sqrt(degrees_of_freedom - 1)
        partial_f = max(partial_f - critical_f, 0.0)
    squared_f = partial_f**2
    return float(0.5 * (sqrt(squared_f**2 + 4 * squared_f) - squared_f))


def _risk_level(robustness_value_significance: float) -> str:
    if robustness_value_significance == 0:
        return "statistically_inconclusive"
    if robustness_value_significance < 0.01:
        return "high"
    if robustness_value_significance < 0.05:
        return "elevated"
    if robustness_value_significance < 0.10:
        return "moderate"
    return "lower"


def _residual_sum_of_squares(
    target: np.ndarray,
    predictors: np.ndarray,
) -> float:
    design = np.column_stack([np.ones(len(target)), predictors])
    coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
    residuals = target - design @ coefficients
    return float(residuals @ residuals)


def _partial_r2(
    target: np.ndarray,
    variable: np.ndarray,
    conditioning: np.ndarray,
) -> float | None:
    reduced_rss = _residual_sum_of_squares(target, conditioning)
    if reduced_rss <= np.finfo(float).eps:
        return None
    full_predictors = np.column_stack([conditioning, variable])
    full_rss = _residual_sum_of_squares(target, full_predictors)
    return float(np.clip((reduced_rss - full_rss) / reduced_rss, 0.0, 1.0))


def _equivalent_robustness_value(
    treatment_partial_r2: float,
    outcome_partial_r2: float,
) -> float:
    treatment_partial_r2 = min(treatment_partial_r2, 1 - 1e-12)
    squared_bias_factor = (
        outcome_partial_r2
        * treatment_partial_r2
        / (1 - treatment_partial_r2)
    )
    return float(
        0.5
        * (
            sqrt(squared_bias_factor**2 + 4 * squared_bias_factor)
            - squared_bias_factor
        )
    )


def build_observed_confounder_benchmark(
    *,
    dataframe: pd.DataFrame,
    treatment: str,
    outcome: str,
    confounders: list[str],
    robustness_value_point_estimate: float,
    robustness_value_significance: float,
) -> dict[str, Any]:
    """使用已观测混杂变量的条件 partial R² 解释 RV 阈值。"""

    if not confounders:
        return {
            "available": False,
            "reason": "No observed confounders were supplied for benchmarking.",
            "confounders": [],
        }
    columns = list(dict.fromkeys([treatment, outcome, *confounders]))
    complete = dataframe[columns].replace([np.inf, -np.inf], np.nan).dropna()
    if len(complete) < len(confounders) + 4:
        return {
            "available": False,
            "reason": "Too few complete rows for observed-confounder benchmarking.",
            "confounders": [],
        }

    treatment_values = complete[treatment].to_numpy(dtype=float)
    outcome_values = complete[outcome].to_numpy(dtype=float)
    rows: list[dict[str, Any]] = []
    for confounder in confounders:
        other_names = [name for name in confounders if name != confounder]
        conditioning_treatment = (
            complete[other_names].to_numpy(dtype=float)
            if other_names
            else np.empty((len(complete), 0))
        )
        conditioning_outcome = np.column_stack(
            [treatment_values, conditioning_treatment]
        )
        variable = complete[confounder].to_numpy(dtype=float)
        treatment_r2 = _partial_r2(
            treatment_values,
            variable,
            conditioning_treatment,
        )
        outcome_r2 = _partial_r2(
            outcome_values,
            variable,
            conditioning_outcome,
        )
        if treatment_r2 is None or outcome_r2 is None:
            continue
        equivalent_rv = _equivalent_robustness_value(treatment_r2, outcome_r2)
        rows.append(
            {
                "confounder": confounder,
                "partial_r2_treatment": treatment_r2,
                "partial_r2_outcome": outcome_r2,
                "equivalent_robustness_value": equivalent_rv,
                "ratio_to_point_rv": (
                    equivalent_rv / robustness_value_point_estimate
                    if robustness_value_point_estimate > 0
                    else None
                ),
                "ratio_to_significance_rv": (
                    equivalent_rv / robustness_value_significance
                    if robustness_value_significance > 0
                    else None
                ),
            }
        )
    rows.sort(
        key=lambda row: float(row["equivalent_robustness_value"]),
        reverse=True,
    )
    if not rows:
        return {
            "available": False,
            "reason": "Observed confounder strengths could not be estimated.",
            "confounders": [],
        }

    strongest = rows[0]
    strongest_rv = float(strongest["equivalent_robustness_value"])
    point_exceeded = (
        strongest_rv >= robustness_value_point_estimate
        if robustness_value_point_estimate > 0
        else None
    )
    significance_exceeded = (
        strongest_rv >= robustness_value_significance
        if robustness_value_significance > 0
        else None
    )
    if point_exceeded:
        assessment = "observed_strength_can_explain_away_point_estimate"
    elif significance_exceeded:
        assessment = "observed_strength_can_remove_significance"
    elif robustness_value_significance <= 0:
        assessment = "estimate_already_statistically_inconclusive"
    else:
        assessment = "stronger_than_observed_benchmark_required"
    return {
        "available": True,
        "method": "leave_one_observed_confounder_partial_r2",
        "strongest_confounder": strongest["confounder"],
        "strongest_equivalent_robustness_value": strongest_rv,
        "strongest_exceeds_point_rv": point_exceeded,
        "strongest_exceeds_significance_rv": significance_exceeded,
        "assessment": assessment,
        "confounders": rows,
        "warnings": [
            "Observed covariates are benchmarks, not upper bounds on the strength "
            "of possible unobserved confounding.",
            "Partial R² values use linear projections, including a linear "
            "probability projection for binary treatment.",
        ],
    }


def build_sensitivity_analysis(
    *,
    estimate: float,
    standard_error: float | None,
    confidence_interval: list[float] | None,
    sample_size: int,
    adjusted_covariate_count: int,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """从效应、标准误和样本量构造未观测混杂敏感性摘要。"""

    if sample_size <= adjusted_covariate_count + 3:
        raise ValueError("Sample size is too small for sensitivity analysis.")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1.")

    inferred_standard_error = False
    resolved_standard_error = standard_error
    if resolved_standard_error is None and confidence_interval is not None:
        if len(confidence_interval) != 2:
            raise ValueError("confidence_interval must contain two values.")
        critical_value = NormalDist().inv_cdf(0.5 + confidence_level / 2)
        resolved_standard_error = (
            float(confidence_interval[1]) - float(confidence_interval[0])
        ) / (2 * critical_value)
        inferred_standard_error = True
    if resolved_standard_error is None or resolved_standard_error <= 0:
        raise ValueError("A positive standard error is required.")

    degrees_of_freedom = sample_size - adjusted_covariate_count - 2
    t_statistic = float(estimate) / float(resolved_standard_error)
    partial_r2 = float(
        t_statistic**2 / (t_statistic**2 + degrees_of_freedom)
    )
    alpha = 1 - confidence_level
    point_rv = robustness_value(
        t_statistic=t_statistic,
        degrees_of_freedom=degrees_of_freedom,
    )
    significance_rv = robustness_value(
        t_statistic=t_statistic,
        degrees_of_freedom=degrees_of_freedom,
        alpha=alpha,
    )
    risk_level = _risk_level(significance_rv)

    warnings = [
        "Partial-R² robustness values are an approximation for non-OLS "
        "estimators or robust/bootstrapped standard errors.",
        "Sensitivity analysis quantifies a hypothetical omitted confounder; "
        "it does not prove that no such confounder exists.",
    ]
    if inferred_standard_error:
        warnings.append(
            "The standard error was inferred from the reported confidence interval."
        )
    if risk_level == "statistically_inconclusive":
        warnings.append(
            "The reported interval is already statistically inconclusive at the "
            "configured confidence level."
        )

    return {
        "available": True,
        "method": "cinelli_hazlett_partial_r2",
        "scope": "approximate_from_effect_standard_error_and_sample_size",
        "t_statistic": t_statistic,
        "degrees_of_freedom": degrees_of_freedom,
        "partial_r2_treatment_outcome": partial_r2,
        "robustness_value_point_estimate": point_rv,
        "robustness_value_significance": significance_rv,
        "unmeasured_confounding_risk": risk_level,
        "risk_thresholds": {
            "high_below": 0.01,
            "elevated_below": 0.05,
            "moderate_below": 0.10,
        },
        "standard_error": float(resolved_standard_error),
        "standard_error_inferred_from_interval": inferred_standard_error,
        "confidence_level": confidence_level,
        "e_value": {
            "computed": False,
            "reason": (
                "The estimator reports an additive ATE/ATT. E-values require a "
                "risk-ratio scale or a justified transformation."
            ),
        },
        "warnings": warnings,
    }
