from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _adjusted_treatment_coefficient(
    outcome: np.ndarray,
    treatment: np.ndarray,
    confounders: np.ndarray,
) -> float:
    design = np.column_stack(
        [np.ones(len(outcome)), treatment, confounders]
    )
    coefficients, *_ = np.linalg.lstsq(design, outcome, rcond=None)
    return float(coefficients[1])


def _linear_projection_fitted_and_residuals(
    treatment: np.ndarray,
    confounders: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    design = np.column_stack([np.ones(len(treatment)), confounders])
    coefficients, *_ = np.linalg.lstsq(design, treatment, rcond=None)
    fitted = design @ coefficients
    return fitted, treatment - fitted


def run_residualized_treatment_permutation(
    *,
    dataframe: pd.DataFrame,
    treatment: str,
    outcome: str,
    confounders: list[str],
    iterations: int = 200,
    random_state: int = 42,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """置换 treatment 线性投影残差，生成快速伪处理分布。"""

    if iterations < 20:
        raise ValueError("iterations must be at least 20.")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1.")

    columns = list(dict.fromkeys([treatment, outcome, *confounders]))
    complete = dataframe[columns].replace([np.inf, -np.inf], np.nan).dropna()
    if len(complete) < max(30, len(confounders) + 5):
        raise ValueError("Too few complete rows for treatment permutation.")
    treatment_values = complete[treatment].to_numpy(dtype=int)
    if sorted(np.unique(treatment_values).tolist()) != [0, 1]:
        raise ValueError("Treatment permutation requires treatment coded as 0/1.")
    outcome_values = complete[outcome].to_numpy(dtype=float)
    confounder_values = (
        complete[confounders].to_numpy(dtype=float)
        if confounders
        else np.empty((len(complete), 0))
    )
    fitted_treatment, treatment_residuals = (
        _linear_projection_fitted_and_residuals(
            treatment_values,
            confounder_values,
        )
    )
    observed = _adjusted_treatment_coefficient(
        outcome_values,
        treatment_values,
        confounder_values,
    )
    rng = np.random.default_rng(random_state)
    placebo_estimates = np.empty(iterations, dtype=float)
    for iteration in range(iterations):
        permuted = fitted_treatment + rng.permutation(treatment_residuals)
        placebo_estimates[iteration] = _adjusted_treatment_coefficient(
            outcome_values,
            permuted,
            confounder_values,
        )

    empirical_p_value = float(
        (1 + np.sum(np.abs(placebo_estimates) >= abs(observed)))
        / (iterations + 1)
    )
    placebo_mean = float(placebo_estimates.mean())
    placebo_sd = float(placebo_estimates.std(ddof=1))
    centering_tolerance = float(
        2 * placebo_sd / np.sqrt(iterations)
        if placebo_sd > 0
        else np.finfo(float).eps
    )
    calibration_passed = abs(placebo_mean) <= centering_tolerance
    return {
        "available": True,
        "method": "residualized_treatment_permutation",
        "diagnostic_statistic": "covariate_adjusted_linear_projection",
        "iterations": iterations,
        "random_state": random_state,
        "pseudo_treatment_type": "linear_fitted_plus_permuted_residual",
        "observed_adjusted_coefficient": observed,
        "placebo_mean": placebo_mean,
        "placebo_standard_deviation": placebo_sd,
        "placebo_quantiles": {
            "q025": float(np.quantile(placebo_estimates, 0.025)),
            "q50": float(np.quantile(placebo_estimates, 0.50)),
            "q975": float(np.quantile(placebo_estimates, 0.975)),
        },
        "empirical_two_sided_p_value": empirical_p_value,
        "monte_carlo_standard_error": float(
            np.sqrt(
                empirical_p_value
                * (1 - empirical_p_value)
                / (iterations + 1)
            )
        ),
        "alpha": alpha,
        "signal_status": (
            "stronger_than_permuted_labels"
            if empirical_p_value <= alpha
            else "not_distinguishable_from_permuted_labels"
        ),
        "calibration_passed": calibration_passed,
        "centering_tolerance": centering_tolerance,
        "warnings": [
            "This is a diagnostic based on a covariate-adjusted linear "
            "projection, not a refit of the primary causal estimator.",
            "The pseudo treatment is continuous after residual permutation and "
            "is used only as a diagnostic statistic.",
            "Residualized permutations are heuristic for observational data and "
            "do not create a randomized experiment.",
            "A non-small empirical p-value can reflect low power and does not "
            "prove absence of a causal effect.",
        ],
    }
