from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from pandas.api.types import is_numeric_dtype

from causalens.schemas import DIDReadinessResult, DIDResult


def _success(result: DIDReadinessResult | DIDResult) -> dict[str, Any]:
    return {"success": True, "result": result.model_dump(), "error": None}


def _error(message: str, **extra: Any) -> dict[str, Any]:
    return {"success": False, "result": None, "error": message, **extra}


def _cell_key(group: int, post: int) -> str:
    return f"group_{group}_post_{post}"


def check_did_readiness(
    *,
    dataset_path: str,
    group_column: str,
    post_column: str,
    outcome: str,
    confounders: list[str] | None = None,
    unit_id_column: str | None = None,
    min_complete_cases: int = 100,
    min_cell_size: int = 20,
) -> dict[str, Any]:
    """检查两期 DiD 的字段、四单元格和面板结构。"""

    try:
        dataframe = pd.read_csv(dataset_path)
        normalized_confounders = list(dict.fromkeys(confounders or []))
        required = list(
            dict.fromkeys(
                [
                    group_column,
                    post_column,
                    outcome,
                    *normalized_confounders,
                    *([unit_id_column] if unit_id_column else []),
                ]
            )
        )
        missing = [column for column in required if column not in dataframe]
        if missing:
            return _error(f"DiD 字段不存在：{missing}")

        blocking: list[str] = []
        warnings: list[str] = []
        for column in [group_column, post_column, outcome, *normalized_confounders]:
            if not is_numeric_dtype(dataframe[column]):
                blocking.append(f"DiD 字段 '{column}' 必须是数值类型。")

        selected = dataframe[required].replace([np.inf, -np.inf], np.nan)
        complete = selected.dropna().copy()
        if len(complete) < min_complete_cases:
            blocking.append(
                f"DiD 完整样本量低于最低要求 {min_complete_cases}。"
            )

        for column, label in (
            (group_column, "处理组标记"),
            (post_column, "post 标记"),
        ):
            values = sorted(complete[column].unique().tolist())
            if values != [0, 1]:
                blocking.append(
                    f"{label} '{column}' 必须严格编码为 0/1；当前值为 {values}。"
                )

        cell_counts: dict[str, int] = {}
        cell_means: dict[str, float] = {}
        if not blocking or len(complete) > 0:
            for group in (0, 1):
                for post in (0, 1):
                    mask = (
                        (complete[group_column] == group)
                        & (complete[post_column] == post)
                    )
                    key = _cell_key(group, post)
                    cell_counts[key] = int(mask.sum())
                    if mask.any() and is_numeric_dtype(complete[outcome]):
                        cell_means[key] = float(complete.loc[mask, outcome].mean())
                    if int(mask.sum()) < min_cell_size:
                        blocking.append(
                            f"DiD 单元格 {key} 的样本量低于 {min_cell_size}。"
                        )

        panel_summary: dict[str, Any] = {}
        if unit_id_column and len(complete) > 0:
            duplicated = complete.duplicated(
                subset=[unit_id_column, post_column],
                keep=False,
            )
            if duplicated.any():
                blocking.append("面板 DiD 要求每个 unit 在每个 pre/post 状态最多一行。")
            group_variation = complete.groupby(unit_id_column)[group_column].nunique()
            if (group_variation > 1).any():
                blocking.append("处理组标记必须在同一 unit 内保持不变。")
            period_counts = complete.groupby(unit_id_column)[post_column].nunique()
            complete_panel_units = int((period_counts == 2).sum())
            total_units = int(period_counts.size)
            panel_summary = {
                "total_units": total_units,
                "complete_two_period_units": complete_panel_units,
            }
            if complete_panel_units != total_units:
                blocking.append("面板 DiD 要求每个 unit 同时具有 pre 和 post 观测。")

        warnings.append(
            "只有一个 pre 时点时无法用数据检验平行趋势；必须依赖研究设计论证。"
        )
        warnings.append(
            "DiD 还要求不存在与处理同步、且只影响处理组的其他冲击。"
        )
        result = DIDReadinessResult(
            ready=not blocking,
            group_column=group_column,
            post_column=post_column,
            outcome=outcome,
            confounders=normalized_confounders,
            unit_id_column=unit_id_column,
            blocking_issues=blocking,
            warnings=warnings,
            sample_summary={
                "total_rows": int(len(dataframe)),
                "complete_cases": int(len(complete)),
                "dropped_rows": int(len(dataframe) - len(complete)),
                "cell_counts": cell_counts,
                "cell_means": cell_means,
                **panel_summary,
            },
            thresholds={
                "min_complete_cases": min_complete_cases,
                "min_cell_size": min_cell_size,
            },
        )
        return _success(result)
    except (FileNotFoundError, pd.errors.EmptyDataError, ValueError) as exc:
        return _error(str(exc))
    except Exception as exc:
        return _error(f"DiD 准备度检查失败：{exc}")


def run_difference_in_differences(
    *,
    dataset_path: str,
    group_column: str,
    post_column: str,
    outcome: str,
    confounders: list[str] | None = None,
    unit_id_column: str | None = None,
) -> dict[str, Any]:
    """估计两期面板或重复横截面的 Difference-in-Differences ATT。"""

    readiness_response = check_did_readiness(
        dataset_path=dataset_path,
        group_column=group_column,
        post_column=post_column,
        outcome=outcome,
        confounders=confounders,
        unit_id_column=unit_id_column,
    )
    if not readiness_response.get("success"):
        return _error(
            "DiD readiness check failed.",
            readiness=readiness_response,
        )
    readiness = readiness_response["result"]
    if readiness.get("ready") is not True:
        return _error(
            "DiD was not executed because readiness did not pass.",
            readiness=readiness,
        )

    try:
        normalized_confounders = list(dict.fromkeys(confounders or []))
        columns = list(
            dict.fromkeys(
                [
                    group_column,
                    post_column,
                    outcome,
                    *normalized_confounders,
                    *([unit_id_column] if unit_id_column else []),
                ]
            )
        )
        data = (
            pd.read_csv(dataset_path)[columns]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
            .copy()
        )
        cell_means = readiness["sample_summary"]["cell_means"]
        unadjusted_did = float(
            (cell_means[_cell_key(1, 1)] - cell_means[_cell_key(1, 0)])
            - (cell_means[_cell_key(0, 1)] - cell_means[_cell_key(0, 0)])
        )

        if unit_id_column:
            outcome_wide = data.pivot(
                index=unit_id_column,
                columns=post_column,
                values=outcome,
            )
            delta_outcome = outcome_wide[1] - outcome_wide[0]
            unit_group = data.groupby(unit_id_column)[group_column].first()
            design = pd.DataFrame({"treated_group": unit_group})
            used_controls: list[str] = []
            for confounder in normalized_confounders:
                control_wide = data.pivot(
                    index=unit_id_column,
                    columns=post_column,
                    values=confounder,
                )
                delta = control_wide[1] - control_wide[0]
                if float(delta.var()) > np.finfo(float).eps:
                    name = f"delta_{confounder}"
                    design[name] = delta
                    used_controls.append(name)
            fitted = sm.OLS(
                delta_outcome,
                sm.add_constant(design, has_constant="add"),
            ).fit(cov_type="HC1")
            effect_term = "treated_group"
            design_mode = "two_period_panel_first_difference"
            sample_size = int(len(design))
        else:
            design = pd.DataFrame(
                {
                    "treated_group": data[group_column].astype(float),
                    "post": data[post_column].astype(float),
                }
            )
            design["treated_x_post"] = (
                design["treated_group"] * design["post"]
            )
            for confounder in normalized_confounders:
                design[confounder] = data[confounder].astype(float)
            fitted = sm.OLS(
                data[outcome].astype(float),
                sm.add_constant(design, has_constant="add"),
            ).fit(cov_type="HC1")
            effect_term = "treated_x_post"
            design_mode = "repeated_cross_section_interaction_ols"
            used_controls = normalized_confounders
            sample_size = int(len(data))

        estimate = float(fitted.params[effect_term])
        standard_error = float(fitted.bse[effect_term])
        interval = fitted.conf_int(alpha=0.05).loc[effect_term]
        result = DIDResult(
            group_column=group_column,
            post_column=post_column,
            outcome=outcome,
            confounders=normalized_confounders,
            unit_id_column=unit_id_column,
            sample_size=sample_size,
            ate=estimate,
            standard_error=standard_error,
            confidence_interval=[float(interval.iloc[0]), float(interval.iloc[1])],
            cell_means={key: float(value) for key, value in cell_means.items()},
            unadjusted_did=unadjusted_did,
            model_configuration={
                "estimator": "DifferenceInDifferences",
                "design_mode": design_mode,
                "effect_term": effect_term,
                "covariance": "HC1",
                "included_controls": used_controls,
            },
            readiness_summary=readiness,
            assumptions=[
                "Parallel trends in the absence of treatment.",
                "No anticipation of treatment before the post period.",
                "No treatment-group-specific concurrent shock.",
                "Stable composition and consistently defined outcomes over time.",
            ],
            warnings=list(readiness["warnings"]),
        )
        return _success(result)
    except (ValueError, KeyError) as exc:
        return _error(str(exc))
    except Exception as exc:
        return _error(f"Difference-in-Differences analysis failed: {exc}")
