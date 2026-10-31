from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from scipy.stats import chi2, norm
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from causalens.schemas import StaggeredDIDReadinessResult, StaggeredDIDResult


def _success(
    result: StaggeredDIDReadinessResult | StaggeredDIDResult,
) -> dict[str, Any]:
    return {"success": True, "result": result.model_dump(), "error": None}


def _error(message: str, **extra: Any) -> dict[str, Any]:
    return {"success": False, "result": None, "error": message, **extra}


def check_staggered_did_readiness(
    *,
    dataset_path: str,
    treatment_column: str,
    time_column: str,
    treatment_start_column: str,
    outcome: str,
    unit_id_column: str,
    confounders: list[str] | None = None,
    min_units_per_group: int = 10,
    min_cohorts: int = 2,
) -> dict[str, Any]:
    """检查错位处理面板、处理起始期和可用对照支持。"""

    try:
        dataframe = pd.read_csv(dataset_path)
        normalized_confounders = list(dict.fromkeys(confounders or []))
        required = list(
            dict.fromkeys(
                [
                    treatment_column,
                    time_column,
                    treatment_start_column,
                    outcome,
                    unit_id_column,
                    *normalized_confounders,
                ]
            )
        )
        missing = [column for column in required if column not in dataframe]
        if missing:
            return _error(f"错位 DiD 字段不存在：{missing}")

        blocking: list[str] = []
        warnings: list[str] = []
        for column in [
            treatment_column,
            time_column,
            treatment_start_column,
            outcome,
            *normalized_confounders,
        ]:
            if not is_numeric_dtype(dataframe[column]):
                blocking.append(f"错位 DiD 字段 '{column}' 必须是数值类型。")

        complete_columns = [
            treatment_column,
            time_column,
            outcome,
            unit_id_column,
            *normalized_confounders,
        ]
        data = dataframe.dropna(subset=complete_columns).copy()
        if normalized_confounders:
            warnings.append(
                "双重稳健调整使用各 cohort 基准期的 confounders；这些字段必须是"
                "处理前变量，不能是中介或处理后变量。"
            )

        if is_numeric_dtype(data[time_column]):
            values = data[time_column].astype(float)
            if not np.allclose(values, np.round(values)):
                blocking.append("time_column 必须使用整数时期编码。")
            else:
                data[time_column] = values.astype(int)
        treatment_values = sorted(data[treatment_column].unique().tolist())
        if treatment_values != [0, 1]:
            blocking.append(
                f"处理状态 '{treatment_column}' 必须严格编码为 0/1；"
                f"当前值为 {treatment_values}。"
            )
        starts = data[treatment_start_column].dropna().astype(float)
        if len(starts) and not np.allclose(starts, np.round(starts)):
            blocking.append("treatment_start_column 必须是整数时期，永不处理单位用空值。")

        if data.duplicated([unit_id_column, time_column]).any():
            blocking.append("每个 unit 在每个时期只能有一行。")
        start_variation = data.groupby(unit_id_column)[treatment_start_column].nunique(
            dropna=False
        )
        if (start_variation > 1).any():
            blocking.append("treatment_start_column 必须在同一 unit 内保持不变。")

        unit_starts = data.groupby(unit_id_column)[treatment_start_column].first()
        cohorts = sorted(unit_starts.dropna().astype(int).unique().tolist())
        cohort_counts = {
            str(cohort): int((unit_starts == cohort).sum()) for cohort in cohorts
        }
        if len(cohorts) < min_cohorts:
            blocking.append(f"至少需要 {min_cohorts} 个不同的处理起始 cohort。")
        for cohort, count in cohort_counts.items():
            if count < min_units_per_group:
                blocking.append(
                    f"处理 cohort {cohort} 的单位数低于 {min_units_per_group}。"
                )

        never_treated_units = int(unit_starts.isna().sum())
        expected = pd.Series(0, index=data.index, dtype=int)
        has_start = data[treatment_start_column].notna()
        expected.loc[has_start] = (
            data.loc[has_start, time_column]
            >= data.loc[has_start, treatment_start_column]
        ).astype(int)
        if treatment_values == [0, 1] and not data[treatment_column].astype(int).equals(expected):
            blocking.append(
                "处理状态必须在 treatment_start_column 起始期及之后为 1，"
                "此前以及永不处理单位为 0。"
            )

        periods = sorted(data[time_column].astype(int).unique().tolist())
        supported_cells = 0
        for cohort in cohorts:
            if cohort - 1 not in periods:
                blocking.append(f"cohort {cohort} 缺少基准期 {cohort - 1}。")
                continue
            for period in periods:
                if period == cohort - 1:
                    continue
                controls = unit_starts[unit_starts.isna() | (unit_starts > max(period, cohort - 1))]
                if len(controls) >= min_units_per_group:
                    supported_cells += 1
            post_controls = unit_starts[
                unit_starts.isna() | (unit_starts > periods[-1])
            ]
            if len(post_controls) < min_units_per_group and never_treated_units < min_units_per_group:
                warnings.append(
                    f"cohort {cohort} 在较晚时期的尚未处理对照可能不足。"
                )

        result = StaggeredDIDReadinessResult(
            ready=not blocking,
            treatment_column=treatment_column,
            time_column=time_column,
            treatment_start_column=treatment_start_column,
            outcome=outcome,
            unit_id_column=unit_id_column,
            blocking_issues=blocking,
            warnings=[
                *warnings,
                "估计使用尚未处理或永不处理单位作为每个 cohort-time 的对照。",
                "预趋势未显著不能证明平行趋势，处理时点也必须满足无预期效应。",
            ],
            sample_summary={
                "total_rows": int(len(dataframe)),
                "analysis_rows": int(len(data)),
                "total_units": int(unit_starts.size),
                "periods": periods,
                "cohorts": cohorts,
                "cohort_counts": cohort_counts,
                "never_treated_units": never_treated_units,
                "supported_group_time_cells": supported_cells,
                "confounders": normalized_confounders,
            },
            thresholds={
                "min_units_per_group": min_units_per_group,
                "min_cohorts": min_cohorts,
            },
        )
        return _success(result)
    except (FileNotFoundError, pd.errors.EmptyDataError, ValueError) as exc:
        return _error(str(exc))
    except Exception as exc:
        return _error(f"错位 DiD 准备度检查失败：{exc}")


def _doubly_robust_att(
    delta: np.ndarray,
    treated: np.ndarray,
    features: np.ndarray,
    *,
    cross_fitting_folds: int,
    random_state: int,
) -> tuple[float, dict[str, float]]:
    """Panel DR-DiD score with normalized treated/control weights."""

    sample_size = len(delta)
    propensity = np.zeros(sample_size, dtype=float)
    control_change = np.zeros(sample_size, dtype=float)
    minimum_class = int(min(treated.sum(), (1 - treated).sum()))
    folds = min(cross_fitting_folds, minimum_class)
    if folds >= 2:
        splitter = StratifiedKFold(
            n_splits=folds,
            shuffle=True,
            random_state=random_state,
        )
        splits = splitter.split(features, treated)
    else:
        indices = np.arange(sample_size)
        splits = [(indices, indices)]

    for train, test in splits:
        propensity_model = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=1.0,
                        max_iter=2000,
                        random_state=random_state,
                    ),
                ),
            ]
        )
        propensity_model.fit(features[train], treated[train])
        propensity[test] = propensity_model.predict_proba(features[test])[:, 1]

        control_train = train[treated[train] == 0]
        outcome_model = Pipeline(
            [("scale", StandardScaler()), ("model", Ridge(alpha=1.0))]
        )
        outcome_model.fit(features[control_train], delta[control_train])
        control_change[test] = outcome_model.predict(features[test])

    clipped = np.clip(propensity, 0.01, 0.99)
    residual = delta - control_change
    treated_residual = float(residual[treated == 1].mean())
    control_weights = (1 - treated) * clipped / (1 - clipped)
    control_weight_sum = float(control_weights.sum())
    if control_weight_sum <= np.finfo(float).eps:
        raise ValueError("Doubly robust control weights have zero support.")
    control_residual = float(np.sum(control_weights * residual) / control_weight_sum)
    estimate = treated_residual - control_residual
    effective_sample_size = float(
        control_weight_sum**2 / np.sum(control_weights**2)
    )
    return estimate, {
        "propensity_min": float(clipped.min()),
        "propensity_max": float(clipped.max()),
        "propensity_clip_fraction": float(
            np.mean((propensity < 0.01) | (propensity > 0.99))
        ),
        "weighted_control_ess": effective_sample_size,
        "cross_fitting_folds": float(folds),
    }


def _estimate_cells(
    data: pd.DataFrame,
    *,
    treatment_start_column: str,
    time_column: str,
    outcome: str,
    unit_id_column: str,
    min_units: int,
    confounders: list[str] | None = None,
    cross_fitting_folds: int = 5,
    random_state: int = 42,
) -> list[dict[str, Any]]:
    normalized_confounders = list(dict.fromkeys(confounders or []))
    unit_starts = data.groupby(unit_id_column)[treatment_start_column].first()
    periods = sorted(data[time_column].astype(int).unique().tolist())
    cohorts = sorted(unit_starts.dropna().astype(int).unique().tolist())
    outcome_wide = data.pivot(index=unit_id_column, columns=time_column, values=outcome)
    cells: list[dict[str, Any]] = []
    for cohort in cohorts:
        baseline = cohort - 1
        if baseline not in outcome_wide.columns:
            continue
        treated_units = unit_starts[unit_starts == cohort].index
        for period in periods:
            if period == baseline or period not in outcome_wide.columns:
                continue
            control_units = unit_starts[
                (unit_starts != cohort)
                & (
                    unit_starts.isna()
                    | (unit_starts > max(period, baseline))
                )
            ].index
            treated_delta = (
                outcome_wide.loc[treated_units, period]
                - outcome_wide.loc[treated_units, baseline]
            ).dropna()
            control_delta = (
                outcome_wide.loc[control_units, period]
                - outcome_wide.loc[control_units, baseline]
            ).dropna()
            if len(treated_delta) < min_units or len(control_delta) < min_units:
                continue
            unadjusted = float(treated_delta.mean() - control_delta.mean())
            diagnostics: dict[str, float] = {}
            if normalized_confounders:
                eligible_units = treated_delta.index.append(control_delta.index)
                baseline_rows = data[data[time_column] == baseline].set_index(
                    unit_id_column
                )
                baseline_features = baseline_rows.loc[
                    eligible_units, normalized_confounders
                ].astype(float)
                delta = pd.concat([treated_delta, control_delta]).loc[
                    eligible_units
                ].to_numpy(dtype=float)
                treated_indicator = np.concatenate(
                    [
                        np.ones(len(treated_delta), dtype=int),
                        np.zeros(len(control_delta), dtype=int),
                    ]
                )
                estimate, diagnostics = _doubly_robust_att(
                    delta,
                    treated_indicator,
                    baseline_features.to_numpy(dtype=float),
                    cross_fitting_folds=cross_fitting_folds,
                    random_state=random_state + int(cohort) * 1009 + int(period),
                )
                adjustment_mode = "cross_fitted_doubly_robust"
            else:
                estimate = unadjusted
                adjustment_mode = "unadjusted_panel_difference"
            cells.append(
                {
                    "cohort": int(cohort),
                    "time": int(period),
                    "event_time": int(period - cohort),
                    "baseline_time": int(baseline),
                    "estimate": float(estimate),
                    "unadjusted_estimate": unadjusted,
                    "treated_units": int(len(treated_delta)),
                    "control_units": int(len(control_delta)),
                    "control_definition": "not_yet_or_never_treated",
                    "adjustment_mode": adjustment_mode,
                    "diagnostics": diagnostics,
                }
            )
    return cells


def _aggregate_cells(cells: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], float]:
    frame = pd.DataFrame(cells)
    dynamic: list[dict[str, Any]] = []
    for event_time, group in frame.groupby("event_time", sort=True):
        weights = group["treated_units"].to_numpy(dtype=float)
        estimate = float(np.average(group["estimate"], weights=weights))
        dynamic.append(
            {
                "event_time": int(event_time),
                "estimate": estimate,
                "cohort_count": int(len(group)),
                "treated_observations": int(weights.sum()),
                "is_reference": False,
            }
        )
    dynamic.append(
        {
            "event_time": -1,
            "estimate": 0.0,
            "cohort_count": 0,
            "treated_observations": 0,
            "is_reference": True,
        }
    )
    dynamic.sort(key=lambda row: row["event_time"])
    post = frame[frame["event_time"] >= 0]
    overall = float(
        np.average(post["estimate"], weights=post["treated_units"])
    )
    return dynamic, overall


def _summarize_nuisance_diagnostics(
    cells: list[dict[str, Any]],
) -> dict[str, float]:
    rows = [cell["diagnostics"] for cell in cells if cell.get("diagnostics")]
    if not rows:
        return {}
    return {
        "propensity_min": min(row["propensity_min"] for row in rows),
        "propensity_max": max(row["propensity_max"] for row in rows),
        "max_propensity_clip_fraction": max(
            row["propensity_clip_fraction"] for row in rows
        ),
        "min_weighted_control_ess": min(
            row["weighted_control_ess"] for row in rows
        ),
    }


def _leave_one_cohort_out(
    cells: list[dict[str, Any]], overall: float
) -> dict[str, Any]:
    frame = pd.DataFrame(cells)
    post = frame[frame["event_time"] >= 0]
    estimates: list[dict[str, Any]] = []
    for cohort in sorted(post["cohort"].unique().tolist()):
        remaining = post[post["cohort"] != cohort]
        if remaining.empty:
            continue
        estimate = float(
            np.average(
                remaining["estimate"],
                weights=remaining["treated_units"],
            )
        )
        estimates.append(
            {
                "excluded_cohort": int(cohort),
                "estimate": estimate,
                "deviation_from_full": float(estimate - overall),
            }
        )
    max_deviation = max(
        (abs(row["deviation_from_full"]) for row in estimates),
        default=0.0,
    )
    return {
        "full_sample_estimate": float(overall),
        "leave_one_cohort_out": estimates,
        "max_absolute_deviation": float(max_deviation),
        "estimate_range": (
            [
                float(min(row["estimate"] for row in estimates)),
                float(max(row["estimate"] for row in estimates)),
            ]
            if estimates
            else [float(overall), float(overall)]
        ),
    }


def _bootstrap_estimates(
    data: pd.DataFrame,
    *,
    treatment_start_column: str,
    time_column: str,
    outcome: str,
    unit_id_column: str,
    target_event_times: list[int],
    iterations: int,
    random_state: int,
    confounders: list[str] | None = None,
) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    units = data[unit_id_column].drop_duplicates().to_numpy()
    rows: list[list[float]] = []
    for _ in range(iterations):
        sampled = rng.choice(units, size=len(units), replace=True)
        pieces = []
        for draw_id, unit in enumerate(sampled):
            piece = data[data[unit_id_column] == unit].copy()
            piece[unit_id_column] = draw_id
            pieces.append(piece)
        draw = pd.concat(pieces, ignore_index=True)
        cells = _estimate_cells(
            draw,
            treatment_start_column=treatment_start_column,
            time_column=time_column,
            outcome=outcome,
            unit_id_column=unit_id_column,
            min_units=2,
            confounders=confounders,
            cross_fitting_folds=1,
            random_state=random_state,
        )
        if not cells:
            continue
        dynamic, overall = _aggregate_cells(cells)
        values = {row["event_time"]: row["estimate"] for row in dynamic}
        if not all(event_time in values for event_time in target_event_times):
            continue
        rows.append([*(values[event_time] for event_time in target_event_times), overall])
    return np.asarray(rows, dtype=float)


def _save_chart(
    dynamic: list[dict[str, Any]], output_dir: str, signature: str
) -> str:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(signature.encode("utf-8")).hexdigest()[:12]
    chart_path = path / f"staggered_did_{digest}.png"
    frame = pd.DataFrame(dynamic).sort_values("event_time")
    figure, axis = plt.subplots(figsize=(8, 4.8))
    axis.errorbar(
        frame["event_time"],
        frame["estimate"],
        yerr=np.vstack(
            [
                frame["estimate"] - frame["confidence_interval"].str[0],
                frame["confidence_interval"].str[1] - frame["estimate"],
            ]
        ),
        fmt="o-",
        capsize=4,
        color="#7c3aed",
    )
    axis.axhline(0, color="#374151", linestyle="--", linewidth=1)
    axis.axvline(-1, color="#dc2626", linestyle=":", linewidth=1)
    axis.set(xlabel="Event time", ylabel="Cohort-aggregated ATT")
    axis.set_title("Staggered-adoption dynamic effects (95% CI)")
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(chart_path, dpi=160)
    plt.close(figure)
    return str(chart_path.resolve())


def run_staggered_did(
    *,
    dataset_path: str,
    treatment_column: str,
    time_column: str,
    treatment_start_column: str,
    outcome: str,
    unit_id_column: str,
    confounders: list[str] | None = None,
    cross_fitting_folds: int = 5,
    bootstrap_iterations: int = 200,
    max_propensity_clip_fraction: float = 0.10,
    min_weighted_control_ess: float = 10.0,
    random_state: int = 42,
    chart_output_dir: str | None = None,
) -> dict[str, Any]:
    """估计错位处理 cohort-time ATT，并按 event time 聚合。"""

    if not 0 <= max_propensity_clip_fraction <= 1:
        return _error("max_propensity_clip_fraction must be between 0 and 1.")
    if min_weighted_control_ess <= 0:
        return _error("min_weighted_control_ess must be positive.")

    readiness_response = check_staggered_did_readiness(
        dataset_path=dataset_path,
        treatment_column=treatment_column,
        time_column=time_column,
        treatment_start_column=treatment_start_column,
        outcome=outcome,
        unit_id_column=unit_id_column,
        confounders=confounders,
    )
    if not readiness_response.get("success"):
        return _error("Staggered DiD readiness check failed.", readiness=readiness_response)
    readiness = readiness_response["result"]
    if readiness.get("ready") is not True:
        return _error(
            "Staggered DiD was not executed because readiness did not pass.",
            readiness=readiness,
        )
    try:
        columns = [
            treatment_column,
            time_column,
            treatment_start_column,
            outcome,
            unit_id_column,
            *(confounders or []),
        ]
        data = pd.read_csv(dataset_path)[columns].dropna(
            subset=[treatment_column, time_column, outcome, unit_id_column]
        )
        data[time_column] = data[time_column].astype(int)
        cells = _estimate_cells(
            data,
            treatment_start_column=treatment_start_column,
            time_column=time_column,
            outcome=outcome,
            unit_id_column=unit_id_column,
            min_units=10,
            confounders=confounders,
            cross_fitting_folds=cross_fitting_folds,
            random_state=random_state,
        )
        if not cells:
            return _error("No supported cohort-time ATT cells were available.")
        nuisance_summary = _summarize_nuisance_diagnostics(cells)
        if nuisance_summary and (
            nuisance_summary["max_propensity_clip_fraction"]
            > max_propensity_clip_fraction
            or nuisance_summary["min_weighted_control_ess"]
            < min_weighted_control_ess
        ):
            return _error(
                "Staggered DiD was blocked because adjusted overlap/ESS "
                "diagnostics did not pass.",
                overlap_diagnostics=nuisance_summary,
                thresholds={
                    "max_propensity_clip_fraction": max_propensity_clip_fraction,
                    "min_weighted_control_ess": min_weighted_control_ess,
                },
            )
        dynamic, overall = _aggregate_cells(cells)
        cohort_sensitivity = _leave_one_cohort_out(cells, overall)
        target_event_times = [
            row["event_time"] for row in dynamic if row["event_time"] != -1
        ]
        bootstrap = _bootstrap_estimates(
            data,
            treatment_start_column=treatment_start_column,
            time_column=time_column,
            outcome=outcome,
            unit_id_column=unit_id_column,
            target_event_times=target_event_times,
            iterations=bootstrap_iterations,
            random_state=random_state,
            confounders=confounders,
        )
        if len(bootstrap) < max(30, bootstrap_iterations // 2):
            return _error("Too few valid unit-bootstrap replications for inference.")
        bootstrap_se = bootstrap.std(axis=0, ddof=1)
        dynamic_lookup = {row["event_time"]: row for row in dynamic}
        for index, event_time in enumerate(target_event_times):
            row = dynamic_lookup[event_time]
            se = float(bootstrap_se[index])
            row["standard_error"] = se
            row["confidence_interval"] = [
                float(row["estimate"] - 1.959963984540054 * se),
                float(row["estimate"] + 1.959963984540054 * se),
            ]
        reference = dynamic_lookup[-1]
        reference["standard_error"] = 0.0
        reference["confidence_interval"] = [0.0, 0.0]

        overall_se = float(bootstrap_se[-1])
        overall_interval = [
            overall - 1.959963984540054 * overall_se,
            overall + 1.959963984540054 * overall_se,
        ]
        lead_times = [event_time for event_time in target_event_times if event_time < -1]
        if lead_times:
            lead_indices = [target_event_times.index(event_time) for event_time in lead_times]
            lead_estimates = np.asarray(
                [dynamic_lookup[event_time]["estimate"] for event_time in lead_times]
            )
            lead_covariance = np.cov(bootstrap[:, lead_indices], rowvar=False)
            lead_covariance = np.atleast_2d(lead_covariance)
            statistic = float(lead_estimates @ np.linalg.pinv(lead_covariance) @ lead_estimates)
            p_value = float(chi2.sf(statistic, df=len(lead_times)))
        else:
            statistic, p_value = float("nan"), float("nan")
        pretrend = {
            "null_hypothesis": "All aggregated pre-treatment lead effects equal zero.",
            "tested_periods": lead_times,
            "statistic": statistic,
            "degrees_of_freedom": len(lead_times),
            "p_value": p_value,
            "significance_level": 0.05,
            "passed": bool(p_value >= 0.05) if lead_times else None,
            "interpretation": (
                "No statistically significant pre-trend was detected; this does not prove parallel trends."
                if lead_times and p_value >= 0.05
                else "The joint pre-trend test rejects zero lead effects; parallel trends is questionable."
                if lead_times
                else "No estimable lead periods were available."
            ),
        }
        if -2 in target_event_times:
            placebo_index = target_event_times.index(-2)
            placebo_estimate = float(-dynamic_lookup[-2]["estimate"])
            placebo_se = float(bootstrap_se[placebo_index])
            placebo_z = (
                placebo_estimate / placebo_se
                if placebo_se > np.finfo(float).eps
                else float("inf")
            )
            placebo_pvalue = float(2 * norm.sf(abs(placebo_z)))
            placebo_test = {
                "pseudo_treatment_shift": -1,
                "comparison_periods": [-2, -1],
                "estimate": placebo_estimate,
                "standard_error": placebo_se,
                "confidence_interval": [
                    placebo_estimate - 1.959963984540054 * placebo_se,
                    placebo_estimate + 1.959963984540054 * placebo_se,
                ],
                "p_value": placebo_pvalue,
                "significance_level": 0.05,
                "passed": placebo_pvalue >= 0.05,
                "interpretation": (
                    "The one-period-early placebo effect was not statistically significant."
                    if placebo_pvalue >= 0.05
                    else "The one-period-early placebo effect was significant; no-anticipation is questionable."
                ),
            }
        else:
            placebo_test = {
                "available": False,
                "reason": "Event time -2 is required for the one-period-early placebo test.",
            }
        chart_path = None
        if chart_output_dir:
            chart_path = _save_chart(
                dynamic,
                chart_output_dir,
                "|".join(
                    [dataset_path, treatment_column, time_column, treatment_start_column, outcome]
                ),
            )
        warnings = list(readiness["warnings"])
        if lead_times and p_value < 0.05:
            warnings.append(
                "联合预趋势检验显著；处理后的 cohort-time ATT 不应直接作强因果解释。"
            )
        if placebo_test.get("passed") is False:
            warnings.append(
                "提前一期的伪处理效应显著；无预期效应假设可能不成立。"
            )
        result = StaggeredDIDResult(
            treatment_column=treatment_column,
            time_column=time_column,
            treatment_start_column=treatment_start_column,
            outcome=outcome,
            unit_id_column=unit_id_column,
            confounders=list(confounders or []),
            sample_size=int(len(data)),
            ate=float(overall),
            standard_error=overall_se,
            confidence_interval=[float(value) for value in overall_interval],
            group_time_effects=cells,
            dynamic_effects=dynamic,
            pretrend_test=pretrend,
            placebo_test=placebo_test,
            cohort_sensitivity=cohort_sensitivity,
            chart_path=chart_path,
            model_configuration={
                "estimator": "StaggeredAdoptionDID",
                "comparison_group": "not_yet_or_never_treated",
                "baseline": "cohort_start_minus_one",
                "aggregation": "treated_unit_weighted",
                "inference": "unit_cluster_bootstrap",
                "bootstrap_iterations": int(len(bootstrap)),
                "random_state": random_state,
                "adjustment": (
                    "cross_fitted_doubly_robust"
                    if confounders
                    else "unadjusted_panel_difference"
                ),
                "confounders": list(confounders or []),
                "cross_fitting_folds": cross_fitting_folds if confounders else 0,
                "nuisance_diagnostics": nuisance_summary,
                "overlap_thresholds": {
                    "max_propensity_clip_fraction": max_propensity_clip_fraction,
                    "min_weighted_control_ess": min_weighted_control_ess,
                },
            },
            readiness_summary=readiness,
            assumptions=[
                "Cohort-specific parallel trends relative to not-yet or never-treated controls.",
                "No anticipation before each cohort treatment start.",
                "No interference between units and no treatment reversal.",
                "Treatment timing and outcome measurement are consistently defined.",
                "Supplied confounders are measured before treatment and correctly specified for at least one nuisance model.",
            ],
            warnings=warnings,
        )
        return _success(result)
    except (ValueError, KeyError) as exc:
        return _error(str(exc))
    except Exception as exc:
        return _error(f"Staggered-adoption DiD analysis failed: {exc}")
