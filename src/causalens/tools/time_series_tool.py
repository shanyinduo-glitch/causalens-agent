from __future__ import annotations

from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from pandas.api.types import is_numeric_dtype
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller

from causalens.schemas import (
    TimeSeriesLocalProjectionResult,
    TimeSeriesReadinessResult,
)


def _error(message: str) -> dict[str, Any]:
    return {"success": False, "result": None, "error": message}


def _load_ordered_frame(
    *,
    dataset_path: str,
    treatment: str,
    outcome: str,
    time_column: str,
    confounders: list[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = pd.read_csv(dataset_path)
    required = list(dict.fromkeys([time_column, treatment, outcome, *confounders]))
    missing = [column for column in required if column not in raw.columns]
    if missing:
        raise ValueError(f"Time-series fields are missing: {missing}")
    for column in [treatment, outcome, *confounders]:
        if not is_numeric_dtype(raw[column]):
            raise ValueError(f"Time-series analysis field {column!r} must be numeric.")

    selected = raw[required].replace([np.inf, -np.inf], np.nan).copy()
    if is_numeric_dtype(selected[time_column]):
        selected["__time_order__"] = pd.to_numeric(
            selected[time_column], errors="coerce"
        )
    else:
        selected["__time_order__"] = pd.to_datetime(
            selected[time_column], errors="coerce"
        )
    invalid_time = int(selected["__time_order__"].isna().sum())
    duplicate_time = int(selected["__time_order__"].duplicated().sum())
    selected = selected.dropna().sort_values("__time_order__").reset_index(drop=True)
    return selected, {
        "total_rows": int(len(raw)),
        "complete_rows": int(len(selected)),
        "dropped_rows": int(len(raw) - len(selected)),
        "invalid_time_rows": invalid_time,
        "duplicate_time_rows": duplicate_time,
        "time_start": (
            str(selected[time_column].iloc[0]) if not selected.empty else None
        ),
        "time_end": (
            str(selected[time_column].iloc[-1]) if not selected.empty else None
        ),
    }


def _stationarity(values: pd.Series) -> dict[str, Any]:
    clean = values.dropna().astype(float)
    if len(clean) < 25 or clean.nunique() < 3:
        return {"available": False, "reason": "insufficient variation or sample size"}
    try:
        statistic, p_value, used_lag, observations, critical, _ = adfuller(
            clean.to_numpy(), autolag="AIC"
        )
    except Exception as exc:
        return {"available": False, "reason": str(exc)}
    return {
        "available": True,
        "test": "augmented_dickey_fuller",
        "statistic": float(statistic),
        "p_value": float(p_value),
        "used_lag": int(used_lag),
        "observations": int(observations),
        "critical_values": {key: float(value) for key, value in critical.items()},
        "reject_unit_root_at_10_percent": bool(p_value < 0.10),
    }


def check_time_series_readiness(
    *,
    dataset_path: str,
    treatment: str,
    outcome: str,
    time_column: str,
    confounders: list[str] | None = None,
    max_horizon: int = 5,
    treatment_lags: int = 1,
    outcome_lags: int = 1,
    min_complete_observations: int = 80,
) -> dict[str, Any]:
    """Validate an ordered, continuous-treatment local-projection contract."""

    try:
        if max_horizon < 0 or max_horizon > 60:
            raise ValueError("max_horizon must be between 0 and 60.")
        if treatment_lags < 1 or outcome_lags < 1:
            raise ValueError("treatment_lags and outcome_lags must be at least 1.")
        normalized = list(dict.fromkeys(confounders or []))
        frame, summary = _load_ordered_frame(
            dataset_path=dataset_path,
            treatment=treatment,
            outcome=outcome,
            time_column=time_column,
            confounders=normalized,
        )
        blocking: list[str] = []
        warnings: list[str] = []
        if summary["duplicate_time_rows"]:
            blocking.append(
                "Time column contains duplicate observations; a single-series design "
                "requires one row per time point."
            )
        required_rows = max(
            min_complete_observations,
            max_horizon + treatment_lags + outcome_lags + len(normalized) + 20,
        )
        if len(frame) < required_rows:
            blocking.append(
                f"Only {len(frame)} ordered complete observations remain; at least "
                f"{required_rows} are required."
            )
        treatment_unique = int(frame[treatment].nunique())
        if treatment_unique < 10:
            blocking.append(
                "Continuous-shock analysis requires at least 10 distinct treatment values."
            )
        if float(frame[treatment].std(ddof=1)) <= np.finfo(float).eps:
            blocking.append("Treatment has no usable time-series variation.")
        stationarity = {
            "treatment": _stationarity(frame[treatment]),
            "outcome": _stationarity(frame[outcome]),
        }
        for name, diagnostic in stationarity.items():
            if diagnostic.get("available") and not diagnostic.get(
                "reject_unit_root_at_10_percent"
            ):
                warnings.append(
                    f"ADF does not reject a unit root for {name}; use changes/returns or "
                    "another defensible transformation before causal interpretation."
                )
        if summary["dropped_rows"]:
            warnings.append(
                "Rows with missing, non-finite, or invalid time values are excluded "
                "without imputation."
            )
        result = TimeSeriesReadinessResult(
            ready=not blocking,
            treatment=treatment,
            outcome=outcome,
            time_column=time_column,
            confounders=normalized,
            sample_summary={
                **summary,
                "treatment_unique_count": treatment_unique,
                "treatment_standard_deviation": (
                    float(frame[treatment].std(ddof=1)) if len(frame) > 1 else None
                ),
            },
            stationarity=stationarity,
            blocking_issues=blocking,
            warnings=warnings,
            thresholds={
                "minimum_complete_observations": required_rows,
                "minimum_distinct_treatment_values": 10,
                "adf_warning_level": 0.10,
            },
        )
        return {"success": True, "result": result.model_dump(), "error": None}
    except (ValueError, KeyError, FileNotFoundError) as exc:
        return _error(str(exc))
    except Exception as exc:
        return _error(f"Time-series readiness failed: {exc}")


def _projection_frame(
    frame: pd.DataFrame,
    *,
    treatment: str,
    outcome: str,
    confounders: list[str],
    horizon: int,
    treatment_lags: int,
    outcome_lags: int,
    control_lag: int,
    cumulative_outcome: bool,
) -> tuple[pd.Series, pd.DataFrame]:
    if cumulative_outcome:
        target = sum(frame[outcome].shift(-step) for step in range(horizon + 1))
    else:
        target = frame[outcome].shift(-horizon)
    regressors: dict[str, pd.Series] = {"shock": frame[treatment]}
    for lag in range(1, treatment_lags + 1):
        regressors[f"treatment_lag_{lag}"] = frame[treatment].shift(lag)
    for lag in range(1, outcome_lags + 1):
        regressors[f"outcome_lag_{lag}"] = frame[outcome].shift(lag)
    for column in confounders:
        regressors[f"control_{column}_lag_{control_lag}"] = frame[column].shift(
            control_lag
        )
    joined = pd.concat(
        [target.rename("target"), pd.DataFrame(regressors)], axis=1
    ).dropna()
    return joined.pop("target"), sm.add_constant(joined, has_constant="add")


def run_time_series_local_projection(
    *,
    dataset_path: str,
    treatment: str,
    outcome: str,
    time_column: str,
    confounders: list[str] | None = None,
    max_horizon: int = 5,
    treatment_lags: int = 1,
    outcome_lags: int = 1,
    control_lag: int = 1,
    cumulative_outcome: bool = True,
    hac_lags: int | None = None,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """Estimate dynamic responses to one unit of an observed continuous shock."""

    try:
        if control_lag < 1:
            raise ValueError("control_lag must be at least 1 to avoid contemporaneous controls.")
        if not 0 < confidence_level < 1:
            raise ValueError("confidence_level must be between 0 and 1.")
        normalized = list(dict.fromkeys(confounders or []))
        readiness_response = check_time_series_readiness(
            dataset_path=dataset_path,
            treatment=treatment,
            outcome=outcome,
            time_column=time_column,
            confounders=normalized,
            max_horizon=max_horizon,
            treatment_lags=treatment_lags,
            outcome_lags=outcome_lags,
        )
        if readiness_response.get("success") is not True:
            return _error(str(readiness_response.get("error")))
        readiness = readiness_response["result"]
        if readiness["ready"] is not True:
            return _error(
                "Time-series readiness did not pass: "
                + "; ".join(readiness["blocking_issues"])
            )
        frame, _ = _load_ordered_frame(
            dataset_path=dataset_path,
            treatment=treatment,
            outcome=outcome,
            time_column=time_column,
            confounders=normalized,
        )
        critical = NormalDist().inv_cdf(0.5 + confidence_level / 2)
        dynamic: list[dict[str, Any]] = []
        residuals_h0: np.ndarray | None = None
        for horizon in range(max_horizon + 1):
            target, design = _projection_frame(
                frame,
                treatment=treatment,
                outcome=outcome,
                confounders=normalized,
                horizon=horizon,
                treatment_lags=treatment_lags,
                outcome_lags=outcome_lags,
                control_lag=control_lag,
                cumulative_outcome=cumulative_outcome,
            )
            selected_hac_lags = int(
                hac_lags
                if hac_lags is not None
                else max(1, round(4 * (len(target) / 100) ** (2 / 9)))
            )
            fitted = sm.OLS(target.astype(float), design.astype(float)).fit(
                cov_type="HAC",
                cov_kwds={"maxlags": selected_hac_lags},
            )
            estimate = float(fitted.params["shock"])
            standard_error = float(fitted.bse["shock"])
            dynamic.append(
                {
                    "horizon": horizon,
                    "estimate": estimate,
                    "standard_error": standard_error,
                    "confidence_interval": [
                        estimate - critical * standard_error,
                        estimate + critical * standard_error,
                    ],
                    "p_value": float(fitted.pvalues["shock"]),
                    "sample_size": int(fitted.nobs),
                    "hac_lags": selected_hac_lags,
                }
            )
            if horizon == 0:
                residuals_h0 = np.asarray(fitted.resid, dtype=float)

        placebo_frame = frame.copy()
        placebo_frame["__future_shock__"] = placebo_frame[treatment].shift(-1)
        target, design = _projection_frame(
            placebo_frame,
            treatment="__future_shock__",
            outcome=outcome,
            confounders=normalized,
            horizon=0,
            treatment_lags=treatment_lags,
            outcome_lags=outcome_lags,
            control_lag=control_lag,
            cumulative_outcome=False,
        )
        placebo_hac = int(
            hac_lags
            if hac_lags is not None
            else max(1, round(4 * (len(target) / 100) ** (2 / 9)))
        )
        placebo_fit = sm.OLS(target.astype(float), design.astype(float)).fit(
            cov_type="HAC", cov_kwds={"maxlags": placebo_hac}
        )
        placebo = {
            "test": "future_treatment_predicts_current_outcome",
            "estimate": float(placebo_fit.params["shock"]),
            "standard_error": float(placebo_fit.bse["shock"]),
            "p_value": float(placebo_fit.pvalues["shock"]),
            "passed": bool(float(placebo_fit.pvalues["shock"]) >= 0.05),
            "interpretation": (
                "Failure is a temporal falsification warning and may reflect reverse "
                "timing, omitted dynamics, or treatment autocorrelation."
            ),
        }
        residual_diagnostics: dict[str, Any] = {}
        if residuals_h0 is not None and len(residuals_h0) >= 20:
            lag = min(10, max(1, len(residuals_h0) // 10))
            ljung = acorr_ljungbox(residuals_h0, lags=[lag], return_df=True)
            residual_diagnostics = {
                "test": "ljung_box",
                "lag": lag,
                "statistic": float(ljung["lb_stat"].iloc[0]),
                "p_value": float(ljung["lb_pvalue"].iloc[0]),
                "residual_autocorrelation_warning": bool(
                    float(ljung["lb_pvalue"].iloc[0]) < 0.05
                ),
            }
        warnings = list(readiness["warnings"])
        if placebo["passed"] is not True:
            warnings.append(
                "A future-treatment temporal placebo is statistically non-zero; "
                "causal timing may be misspecified."
            )
        if residual_diagnostics.get("residual_autocorrelation_warning"):
            warnings.append(
                "Impact-equation residuals retain autocorrelation despite HAC inference; "
                "inspect lag length and model dynamics."
            )
        impact = dynamic[0]
        result = TimeSeriesLocalProjectionResult(
            treatment=treatment,
            outcome=outcome,
            time_column=time_column,
            confounders=normalized,
            sample_size=int(impact["sample_size"]),
            ate=float(impact["estimate"]),
            standard_error=float(impact["standard_error"]),
            confidence_interval=list(impact["confidence_interval"]),
            confidence_level=confidence_level,
            p_value=float(impact["p_value"]),
            dynamic_effects=dynamic,
            stationarity=readiness["stationarity"],
            placebo_test=placebo,
            residual_diagnostics=residual_diagnostics,
            model_configuration={
                "method": "Jorda-style local projections",
                "response": (
                    "cumulative outcome from t through t+h"
                    if cumulative_outcome
                    else "outcome at t+h"
                ),
                "max_horizon": max_horizon,
                "treatment_lags": treatment_lags,
                "outcome_lags": outcome_lags,
                "control_lag": control_lag,
                "controls_are_lagged": True,
                "covariance": "Newey-West HAC",
                "treatment_scale": "one unit in the source column",
            },
            readiness_summary=readiness["sample_summary"],
            assumptions=[
                "After the declared lagged controls, the treatment innovation is not "
                "confounded with future potential outcomes.",
                "The lag specification captures relevant dynamics and the relationship "
                "is stable over the analyzed period.",
                "There are no omitted simultaneous shocks that move treatment and outcome.",
                "HAC inference addresses serial correlation in errors but does not create "
                "causal identification.",
                "The reported response is per one source-unit treatment change and is not "
                "a binary-treatment ATE.",
            ],
            warnings=list(dict.fromkeys(warnings)),
        )
        return {"success": True, "result": result.model_dump(), "error": None}
    except (ValueError, KeyError, FileNotFoundError) as exc:
        return _error(str(exc))
    except Exception as exc:
        return _error(f"Time-series local projection failed: {exc}")
