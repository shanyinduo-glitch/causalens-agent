from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from pandas.api.types import is_numeric_dtype

from causalens.schemas import EventStudyReadinessResult, EventStudyResult


def _success(
    result: EventStudyReadinessResult | EventStudyResult,
) -> dict[str, Any]:
    return {"success": True, "result": result.model_dump(), "error": None}


def _error(message: str, **extra: Any) -> dict[str, Any]:
    return {"success": False, "result": None, "error": message, **extra}


def _period_label(period: int) -> str:
    return f"m{abs(period)}" if period < 0 else f"p{period}"


def _interaction_name(period: int) -> str:
    return f"treated_event_{_period_label(period)}"


def check_event_study_readiness(
    *,
    dataset_path: str,
    group_column: str,
    event_time_column: str,
    outcome: str,
    confounders: list[str] | None = None,
    unit_id_column: str | None = None,
    post_column: str | None = None,
    reference_period: int = -1,
    min_complete_cases: int = 100,
    min_cell_size: int = 10,
) -> dict[str, Any]:
    """检查多期 Event Study 的相对时间、单元格和面板结构。"""

    try:
        dataframe = pd.read_csv(dataset_path)
        normalized_confounders = list(dict.fromkeys(confounders or []))
        required = list(
            dict.fromkeys(
                [
                    group_column,
                    event_time_column,
                    outcome,
                    *normalized_confounders,
                    *([unit_id_column] if unit_id_column else []),
                    *([post_column] if post_column else []),
                ]
            )
        )
        missing = [column for column in required if column not in dataframe]
        if missing:
            return _error(f"Event Study 字段不存在：{missing}")

        blocking: list[str] = []
        warnings: list[str] = []
        numeric_columns = [
            group_column,
            event_time_column,
            outcome,
            *normalized_confounders,
            *([post_column] if post_column else []),
        ]
        for column in numeric_columns:
            if not is_numeric_dtype(dataframe[column]):
                blocking.append(f"Event Study 字段 '{column}' 必须是数值类型。")

        selected = dataframe[required].replace([np.inf, -np.inf], np.nan)
        complete = selected.dropna().copy()
        if len(complete) < min_complete_cases:
            blocking.append(
                f"Event Study 完整样本量低于最低要求 {min_complete_cases}。"
            )

        group_values = sorted(complete[group_column].unique().tolist())
        if group_values != [0, 1]:
            blocking.append(
                f"处理组标记 '{group_column}' 必须严格编码为 0/1；"
                f"当前值为 {group_values}。"
            )

        periods: list[int] = []
        if is_numeric_dtype(complete[event_time_column]):
            raw_periods = complete[event_time_column].astype(float)
            if not np.allclose(raw_periods, np.round(raw_periods)):
                blocking.append("event_time_column 必须使用整数相对时期编码。")
            else:
                periods = sorted(raw_periods.astype(int).unique().tolist())
                complete[event_time_column] = raw_periods.astype(int)
                if reference_period not in periods:
                    blocking.append(
                        f"参考期 {reference_period} 不存在于 event_time_column。"
                    )
                if not any(period < reference_period for period in periods):
                    blocking.append("至少需要一个早于参考期的处理前时期用于预趋势检验。")
                if not any(period >= 0 for period in periods):
                    blocking.append("至少需要一个 event_time >= 0 的处理后时期。")

        cell_counts: dict[str, int] = {}
        if periods and group_values == [0, 1]:
            for group in (0, 1):
                for period in periods:
                    count = int(
                        (
                            (complete[group_column] == group)
                            & (complete[event_time_column] == period)
                        ).sum()
                    )
                    key = f"group_{group}_event_{period}"
                    cell_counts[key] = count
                    if count < min_cell_size:
                        blocking.append(
                            f"Event Study 单元格 {key} 的样本量低于 {min_cell_size}。"
                        )

        if post_column and is_numeric_dtype(complete[post_column]) and periods:
            post_values = sorted(complete[post_column].unique().tolist())
            if post_values != [0, 1]:
                blocking.append(
                    f"post 标记 '{post_column}' 必须严格编码为 0/1；"
                    f"当前值为 {post_values}。"
                )
            expected_post = (complete[event_time_column] >= 0).astype(int)
            if not complete[post_column].astype(int).equals(expected_post):
                blocking.append(
                    "post_column 必须与 event_time >= 0 的定义完全一致。"
                )

        panel_summary: dict[str, Any] = {}
        if unit_id_column and len(complete) > 0 and periods:
            duplicated = complete.duplicated(
                subset=[unit_id_column, event_time_column], keep=False
            )
            if duplicated.any():
                blocking.append("面板 Event Study 要求每个 unit 在每个时期最多一行。")
            group_variation = complete.groupby(unit_id_column)[group_column].nunique()
            if (group_variation > 1).any():
                blocking.append("处理组标记必须在同一 unit 内保持不变。")
            period_counts = complete.groupby(unit_id_column)[event_time_column].nunique()
            balanced_units = int((period_counts == len(periods)).sum())
            total_units = int(period_counts.size)
            panel_summary = {
                "total_units": total_units,
                "balanced_panel_units": balanced_units,
            }
            if balanced_units != total_units:
                warnings.append(
                    "面板并非完全平衡；估计将使用所有完整行，但需评估样本流失机制。"
                )

        warnings.extend(
            [
                "预趋势检验未拒绝零假设并不能证明平行趋势成立。",
                "Event Study 还要求不存在预期效应和处理组特有的同期冲击。",
            ]
        )
        result = EventStudyReadinessResult(
            ready=not blocking,
            group_column=group_column,
            event_time_column=event_time_column,
            outcome=outcome,
            reference_period=reference_period,
            unit_id_column=unit_id_column,
            blocking_issues=blocking,
            warnings=warnings,
            sample_summary={
                "total_rows": int(len(dataframe)),
                "complete_cases": int(len(complete)),
                "dropped_rows": int(len(dataframe) - len(complete)),
                "periods": periods,
                "pre_periods": [period for period in periods if period < 0],
                "post_periods": [period for period in periods if period >= 0],
                "cell_counts": cell_counts,
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
        return _error(f"Event Study 准备度检查失败：{exc}")


def _save_chart(
    dynamic_effects: list[dict[str, Any]],
    *,
    output_dir: str,
    dataset_path: str,
    group_column: str,
    event_time_column: str,
    outcome: str,
    reference_period: int,
) -> str:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(
        "|".join(
            [dataset_path, group_column, event_time_column, outcome]
        ).encode("utf-8")
    ).hexdigest()[:12]
    chart_path = path / f"event_study_{digest}.png"
    periods = np.asarray([item["event_time"] for item in dynamic_effects])
    estimates = np.asarray([item["estimate"] for item in dynamic_effects])
    lower = np.asarray([item["confidence_interval"][0] for item in dynamic_effects])
    upper = np.asarray([item["confidence_interval"][1] for item in dynamic_effects])

    figure, axis = plt.subplots(figsize=(8, 4.8))
    axis.errorbar(
        periods,
        estimates,
        yerr=np.vstack([estimates - lower, upper - estimates]),
        fmt="o-",
        capsize=4,
        color="#2563eb",
    )
    axis.axhline(0, color="#374151", linewidth=1, linestyle="--")
    axis.axvline(reference_period, color="#dc2626", linewidth=1, linestyle=":")
    axis.set_xlabel("Event time")
    axis.set_ylabel("ATT relative to reference period")
    axis.set_title("Event-study dynamic effects (95% CI)")
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(chart_path, dpi=160)
    plt.close(figure)
    return str(chart_path.resolve())


def run_event_study(
    *,
    dataset_path: str,
    group_column: str,
    event_time_column: str,
    outcome: str,
    confounders: list[str] | None = None,
    unit_id_column: str | None = None,
    post_column: str | None = None,
    reference_period: int = -1,
    chart_output_dir: str | None = None,
) -> dict[str, Any]:
    """估计多期 DiD Event Study、联合预趋势检验和平均处理后 ATT。"""

    readiness_response = check_event_study_readiness(
        dataset_path=dataset_path,
        group_column=group_column,
        event_time_column=event_time_column,
        outcome=outcome,
        confounders=confounders,
        unit_id_column=unit_id_column,
        post_column=post_column,
        reference_period=reference_period,
    )
    if not readiness_response.get("success"):
        return _error("Event Study readiness check failed.", readiness=readiness_response)
    readiness = readiness_response["result"]
    if readiness.get("ready") is not True:
        return _error(
            "Event Study was not executed because readiness did not pass.",
            readiness=readiness,
        )

    try:
        normalized_confounders = list(dict.fromkeys(confounders or []))
        columns = list(
            dict.fromkeys(
                [
                    group_column,
                    event_time_column,
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
        data[event_time_column] = data[event_time_column].astype(int)
        periods = [
            int(period)
            for period in readiness["sample_summary"]["periods"]
        ]
        estimated_periods = [
            period for period in periods if period != reference_period
        ]
        design = pd.DataFrame(index=data.index)
        interaction_terms: dict[int, str] = {}
        for period in estimated_periods:
            label = _period_label(period)
            event_dummy = (data[event_time_column] == period).astype(float)
            design[f"event_{label}"] = event_dummy
            interaction = _interaction_name(period)
            design[interaction] = data[group_column].astype(float) * event_dummy
            interaction_terms[period] = interaction
        for confounder in normalized_confounders:
            design[confounder] = data[confounder].astype(float)

        if unit_id_column:
            grouped = design.groupby(data[unit_id_column])
            fitted_design = design - grouped.transform("mean")
            fitted_outcome = data[outcome].astype(float) - data.groupby(
                unit_id_column
            )[outcome].transform("mean").astype(float)
            nonconstant = fitted_design.var() > np.finfo(float).eps
            fitted_design = fitted_design.loc[:, nonconstant]
            fitted = sm.OLS(fitted_outcome, fitted_design).fit(
                cov_type="cluster",
                cov_kwds={"groups": data[unit_id_column]},
            )
            covariance = "clustered_by_unit"
            design_mode = "panel_unit_fixed_effects"
        else:
            design.insert(0, "treated_group", data[group_column].astype(float))
            fitted_design = sm.add_constant(design, has_constant="add")
            fitted = sm.OLS(data[outcome].astype(float), fitted_design).fit(
                cov_type="HC1"
            )
            covariance = "HC1"
            design_mode = "repeated_cross_section_interaction_ols"

        dynamic_effects: list[dict[str, Any]] = []
        for period in periods:
            if period == reference_period:
                estimate, standard_error = 0.0, 0.0
                interval = [0.0, 0.0]
            else:
                term = interaction_terms[period]
                estimate = float(fitted.params[term])
                standard_error = float(fitted.bse[term])
                conf_int = fitted.conf_int(alpha=0.05).loc[term]
                interval = [float(conf_int.iloc[0]), float(conf_int.iloc[1])]
            dynamic_effects.append(
                {
                    "event_time": period,
                    "estimate": estimate,
                    "standard_error": standard_error,
                    "confidence_interval": interval,
                    "is_reference": period == reference_period,
                }
            )

        lead_periods = [period for period in periods if period < reference_period]
        parameter_names = list(fitted.params.index)
        restrictions = np.zeros((len(lead_periods), len(parameter_names)))
        for row, period in enumerate(lead_periods):
            restrictions[row, parameter_names.index(interaction_terms[period])] = 1.0
        wald = fitted.wald_test(restrictions, scalar=True)
        pretrend_pvalue = float(wald.pvalue)
        pretrend_test = {
            "null_hypothesis": "All pre-treatment interaction effects equal zero.",
            "tested_periods": lead_periods,
            "statistic": float(np.asarray(wald.statistic).squeeze()),
            "p_value": pretrend_pvalue,
            "significance_level": 0.05,
            "passed": pretrend_pvalue >= 0.05,
            "interpretation": (
                "No statistically significant pre-trend was detected; this does not prove parallel trends."
                if pretrend_pvalue >= 0.05
                else "The joint pre-trend test rejects zero lead effects; parallel trends is questionable."
            ),
        }

        post_periods = [period for period in periods if period >= 0]
        weights = np.zeros(len(parameter_names))
        for period in post_periods:
            weights[parameter_names.index(interaction_terms[period])] = 1.0 / len(post_periods)
        average_att = float(weights @ fitted.params.to_numpy())
        covariance_matrix = fitted.cov_params().to_numpy()
        average_variance = float(weights @ covariance_matrix @ weights)
        average_se = float(np.sqrt(max(average_variance, 0.0)))
        average_interval = [
            average_att - 1.959963984540054 * average_se,
            average_att + 1.959963984540054 * average_se,
        ]

        chart_path = None
        if chart_output_dir:
            chart_path = _save_chart(
                dynamic_effects,
                output_dir=chart_output_dir,
                dataset_path=dataset_path,
                group_column=group_column,
                event_time_column=event_time_column,
                outcome=outcome,
                reference_period=reference_period,
            )

        warnings = list(readiness["warnings"])
        if pretrend_pvalue < 0.05:
            warnings.append(
                "联合预趋势检验显著；不应把处理后系数直接解释为可信的因果效应。"
            )
        result = EventStudyResult(
            group_column=group_column,
            event_time_column=event_time_column,
            outcome=outcome,
            confounders=normalized_confounders,
            unit_id_column=unit_id_column,
            reference_period=reference_period,
            sample_size=int(len(data)),
            ate=average_att,
            standard_error=average_se,
            confidence_interval=[float(value) for value in average_interval],
            dynamic_effects=dynamic_effects,
            pretrend_test=pretrend_test,
            chart_path=chart_path,
            model_configuration={
                "estimator": "EventStudy",
                "design_mode": design_mode,
                "covariance": covariance,
                "reference_period": reference_period,
                "post_aggregation": "equal_weight_average",
                "included_controls": normalized_confounders,
            },
            readiness_summary=readiness,
            assumptions=[
                "Parallel trends in the absence of treatment.",
                "No anticipation before treatment starts.",
                "No treatment-group-specific concurrent shock.",
                "Stable composition and consistently defined outcomes over time.",
            ],
            warnings=warnings,
        )
        return _success(result)
    except (ValueError, KeyError) as exc:
        return _error(str(exc))
    except Exception as exc:
        return _error(f"Event Study analysis failed: {exc}")
