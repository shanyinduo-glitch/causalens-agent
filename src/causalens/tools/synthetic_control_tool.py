from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pydantic import ValidationError
from scipy.optimize import minimize

from causalens.schemas import (
    SyntheticControlReadinessResult,
    SyntheticControlResult,
)


def _success(
    result: SyntheticControlReadinessResult | SyntheticControlResult,
) -> dict[str, Any]:
    return {"success": True, "result": result.model_dump(), "error": None}


def _error(message: str) -> dict[str, Any]:
    return {"success": False, "result": None, "error": message}


def _load_panel(
    *,
    dataset_path: str,
    treatment: str,
    outcome: str,
    unit_id_column: str,
    time_column: str,
    covariates: list[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if len(covariates) != len(set(covariates)):
        raise ValueError("covariates contains duplicates.")
    core = {treatment, outcome, unit_id_column, time_column}
    if len(core) != 4:
        raise ValueError(
            "Synthetic-control treatment, outcome, unit ID, and time columns "
            "must be distinct."
        )
    overlap = sorted(core & set(covariates))
    if overlap:
        raise ValueError(
            "Synthetic-control covariates must be distinct from core columns: "
            f"{overlap}"
        )
    raw = pd.read_csv(dataset_path)
    requested = [unit_id_column, time_column, treatment, outcome, *covariates]
    missing = sorted(set(requested) - set(raw.columns))
    if missing:
        raise ValueError(f"Synthetic-control columns not found in CSV: {missing}")
    selected = raw[requested].replace([np.inf, -np.inf], np.nan)
    numeric = [time_column, treatment, outcome, *covariates]
    non_numeric = [
        column
        for column in numeric
        if not pd.api.types.is_numeric_dtype(selected[column])
    ]
    if non_numeric:
        raise ValueError(
            "Synthetic control requires numeric time, treatment, outcome, and "
            f"covariates: {non_numeric}"
        )
    complete = selected.dropna().copy()
    if complete.empty:
        raise ValueError("No complete observations remain for synthetic control.")
    duplicate_count = int(
        complete.duplicated([unit_id_column, time_column], keep=False).sum()
    )
    if duplicate_count:
        raise ValueError(
            "Synthetic control requires one row per unit-time pair; found "
            f"{duplicate_count} duplicated rows."
        )
    complete[time_column] = complete[time_column].astype(float)
    complete[treatment] = complete[treatment].astype(float)
    complete[outcome] = complete[outcome].astype(float)
    for covariate in covariates:
        complete[covariate] = complete[covariate].astype(float)
    return complete, {
        "total_rows": int(len(raw)),
        "complete_rows": int(len(complete)),
        "dropped_rows": int(len(raw) - len(complete)),
        "complete_case_fraction": float(len(complete) / len(raw)),
    }


def _resolve_design(
    *,
    frame: pd.DataFrame,
    treatment: str,
    unit_id_column: str,
    time_column: str,
    intervention_time: float,
    treated_unit: str | int | float | None,
) -> tuple[Any, list[Any], list[float], list[float]]:
    observed_treatment = sorted(frame[treatment].drop_duplicates().tolist())
    if observed_treatment != [0.0, 1.0]:
        raise ValueError(
            "Synthetic-control treatment must be coded as 0 and 1; observed "
            f"values: {observed_treatment}."
        )
    treated_candidates = frame.loc[
        frame[treatment] == 1.0, unit_id_column
    ].drop_duplicates().tolist()
    if treated_unit is None:
        if len(treated_candidates) != 1:
            raise ValueError(
                "Synthetic control currently requires exactly one treated unit; "
                f"found {len(treated_candidates)}."
            )
        resolved_treated = treated_candidates[0]
    else:
        matches = [
            value
            for value in frame[unit_id_column].drop_duplicates().tolist()
            if str(value) == str(treated_unit)
        ]
        if len(matches) != 1:
            raise ValueError(f"treated_unit {treated_unit!r} was not uniquely found.")
        resolved_treated = matches[0]
        if set(treated_candidates) != {resolved_treated}:
            raise ValueError(
                "The declared treated_unit is inconsistent with the treatment column."
            )

    times = sorted(float(value) for value in frame[time_column].unique())
    pre_times = [value for value in times if value < intervention_time]
    post_times = [value for value in times if value >= intervention_time]
    if not pre_times or not post_times:
        raise ValueError(
            "intervention_time must leave at least one observed pre and post period."
        )
    treated_rows = frame[frame[unit_id_column] == resolved_treated]
    treated_pre = treated_rows[treated_rows[time_column] < intervention_time]
    treated_post = treated_rows[treated_rows[time_column] >= intervention_time]
    if not treated_pre.empty and (treated_pre[treatment] != 0).any():
        raise ValueError("The treated unit is already treated before intervention_time.")
    if treated_post.empty or (treated_post[treatment] != 1).any():
        raise ValueError(
            "The treated unit must be treated in every observed post period."
        )
    donor_rows = frame[frame[unit_id_column] != resolved_treated]
    invalid_donors = donor_rows.groupby(unit_id_column)[treatment].max()
    invalid_donors = invalid_donors[invalid_donors > 0].index.tolist()
    if invalid_donors:
        raise ValueError(
            "Donor units must remain untreated throughout the panel: "
            f"{[str(value) for value in invalid_donors]}"
        )
    donors = donor_rows[unit_id_column].drop_duplicates().tolist()
    return resolved_treated, donors, pre_times, post_times


def _balanced_outcomes(
    *,
    frame: pd.DataFrame,
    outcome: str,
    unit_id_column: str,
    time_column: str,
    units: list[Any],
    times: list[float],
) -> pd.DataFrame:
    pivot = frame.pivot(index=time_column, columns=unit_id_column, values=outcome)
    available_units = [
        unit
        for unit in units
        if unit in pivot.columns and pivot.loc[times, unit].notna().all()
    ]
    return pivot.loc[times, available_units]


def _predictor_matrix(
    *,
    frame: pd.DataFrame,
    outcome_pivot: pd.DataFrame,
    units: list[Any],
    pre_times: list[float],
    covariates: list[str],
    unit_id_column: str,
    time_column: str,
    intervention_time: float,
) -> np.ndarray:
    blocks = [outcome_pivot.loc[pre_times, units].to_numpy(dtype=float)]
    if covariates:
        pre = frame[frame[time_column] < intervention_time]
        means = pre.groupby(unit_id_column)[covariates].mean()
        if means.loc[units].isna().any().any():
            raise ValueError("Covariate means are missing for at least one donor unit.")
        blocks.append(means.loc[units].to_numpy(dtype=float).T)
    predictors = np.vstack(blocks)
    row_scale = np.std(predictors, axis=1, ddof=0)
    row_scale[row_scale <= np.finfo(float).eps] = 1.0
    return predictors / row_scale[:, None]


def _fit_weights(target: np.ndarray, donors: np.ndarray) -> np.ndarray:
    donor_count = donors.shape[1]
    if donor_count < 2:
        raise ValueError("At least two eligible donor units are required.")

    def objective(weights: np.ndarray) -> float:
        residual = target - donors @ weights
        return float(residual @ residual + 1e-10 * (weights @ weights))

    fitted = minimize(
        objective,
        np.full(donor_count, 1.0 / donor_count),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * donor_count,
        constraints={"type": "eq", "fun": lambda weights: weights.sum() - 1.0},
        options={"maxiter": 2000, "ftol": 1e-12},
    )
    if not fitted.success:
        raise ValueError(f"Synthetic-control weight optimization failed: {fitted.message}")
    weights = np.clip(np.asarray(fitted.x, dtype=float), 0.0, 1.0)
    return weights / weights.sum()


def _fit_unit(
    *,
    frame: pd.DataFrame,
    outcome_pivot: pd.DataFrame,
    target_unit: Any,
    donor_units: list[Any],
    pre_times: list[float],
    post_times: list[float],
    covariates: list[str],
    unit_id_column: str,
    time_column: str,
    intervention_time: float,
) -> dict[str, Any]:
    units = [target_unit, *donor_units]
    predictors = _predictor_matrix(
        frame=frame,
        outcome_pivot=outcome_pivot,
        units=units,
        pre_times=pre_times,
        covariates=covariates,
        unit_id_column=unit_id_column,
        time_column=time_column,
        intervention_time=intervention_time,
    )
    weights = _fit_weights(predictors[:, 0], predictors[:, 1:])
    target_path = outcome_pivot.loc[[*pre_times, *post_times], target_unit].to_numpy(
        dtype=float
    )
    donor_path = outcome_pivot.loc[
        [*pre_times, *post_times], donor_units
    ].to_numpy(dtype=float)
    synthetic_path = donor_path @ weights
    gaps = target_path - synthetic_path
    pre_count = len(pre_times)
    pre_rmspe = float(np.sqrt(np.mean(np.square(gaps[:pre_count]))))
    post_rmspe = float(np.sqrt(np.mean(np.square(gaps[pre_count:]))))
    ratio = float(post_rmspe / max(pre_rmspe, 1e-12))
    return {
        "weights": weights,
        "target_path": target_path,
        "synthetic_path": synthetic_path,
        "gaps": gaps,
        "att": float(np.mean(gaps[pre_count:])),
        "pre_rmspe": pre_rmspe,
        "post_rmspe": post_rmspe,
        "rmspe_ratio": ratio,
    }


def check_synthetic_control_readiness(
    *,
    dataset_path: str,
    treatment: str,
    outcome: str,
    unit_id_column: str,
    time_column: str,
    intervention_time: float,
    covariates: list[str] | None = None,
    treated_unit: str | int | float | None = None,
) -> dict[str, Any]:
    try:
        normalized_covariates = list(covariates or [])
        frame, sample_summary = _load_panel(
            dataset_path=dataset_path,
            treatment=treatment,
            outcome=outcome,
            unit_id_column=unit_id_column,
            time_column=time_column,
            covariates=normalized_covariates,
        )
        resolved, donors, pre_times, post_times = _resolve_design(
            frame=frame,
            treatment=treatment,
            unit_id_column=unit_id_column,
            time_column=time_column,
            intervention_time=float(intervention_time),
            treated_unit=treated_unit,
        )
        all_units = [resolved, *donors]
        pivot = _balanced_outcomes(
            frame=frame,
            outcome=outcome,
            unit_id_column=unit_id_column,
            time_column=time_column,
            units=all_units,
            times=[*pre_times, *post_times],
        )
        eligible = list(pivot.columns)
        eligible_donors = [unit for unit in donors if unit in eligible]
        blocking: list[str] = []
        warnings: list[str] = []
        if resolved not in eligible:
            blocking.append("The treated unit lacks a complete outcome trajectory.")
        if len(pre_times) < 5:
            blocking.append("Synthetic control requires at least five pre periods.")
        if len(post_times) < 1:
            blocking.append("Synthetic control requires at least one post period.")
        if len(eligible_donors) < 3:
            blocking.append(
                "Synthetic control requires at least three complete untreated donors."
            )
        dropped_donors = len(donors) - len(eligible_donors)
        if dropped_donors:
            warnings.append(
                f"{dropped_donors} donor units were excluded because their panel "
                "trajectory was incomplete."
            )
        if len(pre_times) < 10:
            warnings.append(
                "The pre-treatment period is short; pre-fit and placebo inference "
                "may be unstable."
            )
        sample_summary.update(
            {
                "unit_count": int(frame[unit_id_column].nunique()),
                "treated_unit": str(resolved),
                "eligible_donor_count": len(eligible_donors),
                "pre_period_count": len(pre_times),
                "post_period_count": len(post_times),
                "pre_times": pre_times,
                "post_times": post_times,
            }
        )
        result = SyntheticControlReadinessResult(
            ready=not blocking,
            treatment=treatment,
            outcome=outcome,
            unit_id_column=unit_id_column,
            time_column=time_column,
            intervention_time=float(intervention_time),
            treated_unit=str(resolved),
            covariates=normalized_covariates,
            blocking_issues=blocking,
            warnings=warnings,
            sample_summary=sample_summary,
            thresholds={"minimum_pre_periods": 5, "minimum_donors": 3},
        )
        return _success(result)
    except (ValueError, KeyError, ValidationError) as exc:
        return _error(str(exc))
    except Exception as exc:
        return _error(f"Synthetic-control readiness failed: {exc}")


def run_synthetic_control(
    *,
    dataset_path: str,
    treatment: str,
    outcome: str,
    unit_id_column: str,
    time_column: str,
    intervention_time: float,
    covariates: list[str] | None = None,
    treated_unit: str | int | float | None = None,
) -> dict[str, Any]:
    try:
        normalized_covariates = list(covariates or [])
        readiness_response = check_synthetic_control_readiness(
            dataset_path=dataset_path,
            treatment=treatment,
            outcome=outcome,
            unit_id_column=unit_id_column,
            time_column=time_column,
            intervention_time=intervention_time,
            covariates=normalized_covariates,
            treated_unit=treated_unit,
        )
        if readiness_response.get("success") is not True:
            return readiness_response
        readiness = readiness_response["result"]
        if readiness["ready"] is not True:
            return _error(
                "Synthetic control was not executed because readiness did not pass: "
                + "; ".join(readiness["blocking_issues"])
            )
        frame, _ = _load_panel(
            dataset_path=dataset_path,
            treatment=treatment,
            outcome=outcome,
            unit_id_column=unit_id_column,
            time_column=time_column,
            covariates=normalized_covariates,
        )
        resolved, donors, pre_times, post_times = _resolve_design(
            frame=frame,
            treatment=treatment,
            unit_id_column=unit_id_column,
            time_column=time_column,
            intervention_time=float(intervention_time),
            treated_unit=treated_unit,
        )
        times = [*pre_times, *post_times]
        pivot = _balanced_outcomes(
            frame=frame,
            outcome=outcome,
            unit_id_column=unit_id_column,
            time_column=time_column,
            units=[resolved, *donors],
            times=times,
        )
        donors = [unit for unit in donors if unit in pivot.columns]
        primary = _fit_unit(
            frame=frame,
            outcome_pivot=pivot,
            target_unit=resolved,
            donor_units=donors,
            pre_times=pre_times,
            post_times=post_times,
            covariates=normalized_covariates,
            unit_id_column=unit_id_column,
            time_column=time_column,
            intervention_time=float(intervention_time),
        )
        placebo_rows: list[dict[str, Any]] = []
        for placebo_unit in donors:
            placebo_pool = [unit for unit in donors if unit != placebo_unit]
            if len(placebo_pool) < 2:
                continue
            placebo = _fit_unit(
                frame=frame,
                outcome_pivot=pivot,
                target_unit=placebo_unit,
                donor_units=placebo_pool,
                pre_times=pre_times,
                post_times=post_times,
                covariates=normalized_covariates,
                unit_id_column=unit_id_column,
                time_column=time_column,
                intervention_time=float(intervention_time),
            )
            placebo_rows.append(
                {
                    "unit": str(placebo_unit),
                    "att": placebo["att"],
                    "pre_rmspe": placebo["pre_rmspe"],
                    "post_rmspe": placebo["post_rmspe"],
                    "rmspe_ratio": placebo["rmspe_ratio"],
                }
            )
        placebo_atts = np.asarray([row["att"] for row in placebo_rows], dtype=float)
        standard_error = float(np.std(placebo_atts, ddof=1))
        if not np.isfinite(standard_error) or standard_error <= 0:
            standard_error = float(np.std(primary["gaps"][len(pre_times) :], ddof=1))
        if not np.isfinite(standard_error) or standard_error <= 0:
            standard_error = float(np.finfo(float).eps)
        p_value = float(
            (1 + sum(row["rmspe_ratio"] >= primary["rmspe_ratio"] for row in placebo_rows))
            / (1 + len(placebo_rows))
        )
        interval = [
            float(primary["att"] - 1.96 * standard_error),
            float(primary["att"] + 1.96 * standard_error),
        ]
        trajectories = [
            {
                "time": time,
                "treated_outcome": float(primary["target_path"][index]),
                "synthetic_outcome": float(primary["synthetic_path"][index]),
                "gap": float(primary["gaps"][index]),
                "post": bool(time >= intervention_time),
            }
            for index, time in enumerate(times)
        ]
        warnings = list(readiness["warnings"])
        if primary["pre_rmspe"] > np.std(primary["target_path"][: len(pre_times)]):
            warnings.append(
                "Pre-treatment fit is weak relative to treated-unit outcome variation."
            )
        warnings.append(
            "The reported normal interval uses the dispersion of in-space placebo "
            "effects and should be interpreted as approximate randomization inference."
        )
        result = SyntheticControlResult(
            treatment=treatment,
            outcome=outcome,
            unit_id_column=unit_id_column,
            time_column=time_column,
            intervention_time=float(intervention_time),
            treated_unit=str(resolved),
            covariates=normalized_covariates,
            sample_size=int(len(frame)),
            ate=float(primary["att"]),
            standard_error=standard_error,
            confidence_interval=interval,
            p_value=p_value,
            donor_weights=[
                {"unit": str(unit), "weight": float(weight)}
                for unit, weight in zip(donors, primary["weights"])
                if weight > 1e-8
            ],
            trajectories=trajectories,
            fit_diagnostics={
                "pre_rmspe": primary["pre_rmspe"],
                "post_rmspe": primary["post_rmspe"],
                "post_pre_rmspe_ratio": primary["rmspe_ratio"],
            },
            placebo_inference={
                "method": "in_space_placebo_rmspe_ratio",
                "placebo_count": len(placebo_rows),
                "p_value": p_value,
                "results": placebo_rows,
                "interpretation": (
                    "The p-value ranks the treated unit's post/pre RMSPE ratio "
                    "against donor-unit placebo ratios."
                ),
            },
            model_configuration={
                "weight_constraints": "nonnegative_weights_sum_to_one",
                "predictors": "pre_outcomes_and_pre_covariate_means",
                "optimizer": "SLSQP",
            },
            readiness_summary=readiness,
            assumptions=[
                "No interference or spillovers between the treated unit and donors.",
                "Donor units are not affected by the intervention.",
                "A stable convex combination reproduces the untreated trajectory.",
                "No treated-unit-specific shock coincides with intervention_time.",
            ],
            warnings=warnings,
        )
        return _success(result)
    except (ValueError, KeyError, ValidationError) as exc:
        return _error(str(exc))
    except Exception as exc:
        return _error(f"Synthetic-control estimation failed: {exc}")
