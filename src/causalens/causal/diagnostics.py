from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class BalanceDiagnostic(BaseModel):
    variable: str
    smd_before: float | None
    smd_after: float | None


class CausalDiagnosticsResult(BaseModel):
    analysis_type: str = "causal_diagnostics"
    passed: bool
    sample_size: int
    treatment: str
    confounders: list[str]
    propensity_summary: dict[str, Any] = Field(default_factory=dict)
    weight_summary: dict[str, Any] = Field(default_factory=dict)
    balance: list[BalanceDiagnostic] = Field(default_factory=list)
    blocking_issues: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    thresholds: dict[str, float] = Field(default_factory=dict)


def _success(result: CausalDiagnosticsResult) -> dict[str, Any]:
    return {"success": True, "result": result.model_dump(), "error": None}


def _error(message: str) -> dict[str, Any]:
    return {"success": False, "result": None, "error": message}


def _effective_sample_size(weights: np.ndarray) -> float:
    denominator = float(np.square(weights).sum())
    if denominator <= 0:
        return 0.0
    return float(weights.sum() ** 2 / denominator)


def _weighted_mean_and_variance(
    values: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, float]:
    mean = float(np.average(values, weights=weights))
    variance = float(
        np.average(np.square(values - mean), weights=weights)
    )
    return mean, variance


def _standardized_mean_difference(
    treated: np.ndarray,
    control: np.ndarray,
    treated_weights: np.ndarray | None = None,
    control_weights: np.ndarray | None = None,
) -> float | None:
    if treated_weights is None:
        treated_mean = float(np.mean(treated))
        treated_variance = float(np.var(treated, ddof=1))
    else:
        treated_mean, treated_variance = _weighted_mean_and_variance(
            treated,
            treated_weights,
        )

    if control_weights is None:
        control_mean = float(np.mean(control))
        control_variance = float(np.var(control, ddof=1))
    else:
        control_mean, control_variance = _weighted_mean_and_variance(
            control,
            control_weights,
        )

    pooled_scale = float(
        np.sqrt(max((treated_variance + control_variance) / 2.0, 0.0))
    )
    difference = treated_mean - control_mean
    if pooled_scale <= np.finfo(float).eps:
        return 0.0 if abs(difference) <= np.finfo(float).eps else None
    return float(difference / pooled_scale)


def run_causal_diagnostics(
    *,
    dataset_path: str,
    treatment: str,
    outcome: str,
    confounders: list[str],
    random_state: int = 42,
    max_folds: int = 5,
    extreme_propensity_threshold: float = 0.05,
    balance_threshold: float = 0.10,
    min_common_support_fraction: float = 0.10,
) -> dict[str, Any]:
    """执行估计前 overlap、权重与协变量平衡诊断。"""

    try:
        if not 0 < extreme_propensity_threshold < 0.5:
            raise ValueError("extreme_propensity_threshold 必须在 0 和 0.5 之间。")
        if balance_threshold <= 0:
            raise ValueError("balance_threshold 必须大于 0。")
        if not 0 < min_common_support_fraction <= 1:
            raise ValueError("min_common_support_fraction 必须在 0 和 1 之间。")

        selected_columns = [treatment, outcome, *confounders]
        dataframe = pd.read_csv(dataset_path)[selected_columns]
        complete = (
            dataframe.replace([np.inf, -np.inf], np.nan).dropna().copy()
        )
        if complete.empty:
            return _error("没有可用于因果诊断的完整样本。")

        treatment_values = sorted(complete[treatment].unique().tolist())
        if treatment_values != [0, 1]:
            return _error(
                "因果诊断当前只支持编码为 0/1 的二元 treatment。"
            )
        if not confounders:
            return _error("因果诊断至少需要一个 confounder。")

        features = complete[confounders].to_numpy(dtype=float)
        treatment_array = complete[treatment].to_numpy(dtype=int)
        group_counts = np.bincount(treatment_array, minlength=2)
        folds = int(min(max_folds, group_counts.min()))
        if folds < 2:
            return _error("处理组样本不足，无法执行交叉拟合倾向得分诊断。")

        model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(max_iter=2000)),
            ]
        )
        splitter = StratifiedKFold(
            n_splits=folds,
            shuffle=True,
            random_state=random_state,
        )
        propensity = cross_val_predict(
            model,
            features,
            treatment_array,
            cv=splitter,
            method="predict_proba",
        )[:, 1]

        treated_mask = treatment_array == 1
        control_mask = ~treated_mask
        treated_scores = propensity[treated_mask]
        control_scores = propensity[control_mask]
        support_lower = float(
            max(treated_scores.min(), control_scores.min())
        )
        support_upper = float(
            min(treated_scores.max(), control_scores.max())
        )
        has_common_support = support_lower < support_upper

        if has_common_support:
            treated_support_fraction = float(
                np.mean(
                    (treated_scores >= support_lower)
                    & (treated_scores <= support_upper)
                )
            )
            control_support_fraction = float(
                np.mean(
                    (control_scores >= support_lower)
                    & (control_scores <= support_upper)
                )
            )
        else:
            treated_support_fraction = 0.0
            control_support_fraction = 0.0

        extreme_mask = (
            (propensity < extreme_propensity_threshold)
            | (propensity > 1 - extreme_propensity_threshold)
        )
        extreme_fraction = float(np.mean(extreme_mask))

        clipping = max(extreme_propensity_threshold / 10.0, 1e-6)
        clipped_propensity = np.clip(propensity, clipping, 1 - clipping)
        weights = np.where(
            treated_mask,
            1.0 / clipped_propensity,
            1.0 / (1.0 - clipped_propensity),
        )

        balance_rows: list[BalanceDiagnostic] = []
        for index, variable in enumerate(confounders):
            values = features[:, index]
            balance_rows.append(
                BalanceDiagnostic(
                    variable=variable,
                    smd_before=_standardized_mean_difference(
                        values[treated_mask],
                        values[control_mask],
                    ),
                    smd_after=_standardized_mean_difference(
                        values[treated_mask],
                        values[control_mask],
                        weights[treated_mask],
                        weights[control_mask],
                    ),
                )
            )

        blocking_issues: list[str] = []
        warnings: list[str] = []
        if not has_common_support:
            blocking_issues.append("处理组与对照组的倾向得分没有共同支持区间。")
        elif min(treated_support_fraction, control_support_fraction) < (
            min_common_support_fraction
        ):
            blocking_issues.append(
                "至少一个处理组落在共同支持区间内的样本比例过低。"
            )

        if extreme_fraction > 0.05:
            warnings.append(
                "超过 5% 的样本具有极端倾向得分，positivity 可能较弱。"
            )

        finite_after_smd = [
            abs(row.smd_after)
            for row in balance_rows
            if row.smd_after is not None
        ]
        max_abs_smd_after = max(finite_after_smd, default=0.0)
        if any(row.smd_after is None for row in balance_rows):
            warnings.append("至少一个协变量的标准化均值差无法稳定计算。")
        if max_abs_smd_after > balance_threshold:
            warnings.append(
                "IPW 后仍有协变量的绝对标准化均值差超过阈值。"
            )

        effective_sample_size = _effective_sample_size(weights)
        ess_ratio = float(effective_sample_size / len(complete))
        if ess_ratio < 0.5:
            warnings.append("加权后的有效样本量低于完整样本量的 50%。")

        result = CausalDiagnosticsResult(
            passed=not blocking_issues,
            sample_size=int(len(complete)),
            treatment=treatment,
            confounders=list(confounders),
            propensity_summary={
                "minimum": float(propensity.min()),
                "maximum": float(propensity.max()),
                "mean": float(propensity.mean()),
                "extreme_fraction": extreme_fraction,
                "common_support": [support_lower, support_upper],
                "has_common_support": has_common_support,
                "treated_common_support_fraction": treated_support_fraction,
                "control_common_support_fraction": control_support_fraction,
                "cross_fitting_folds": folds,
            },
            weight_summary={
                "minimum": float(weights.min()),
                "maximum": float(weights.max()),
                "mean": float(weights.mean()),
                "p99": float(np.quantile(weights, 0.99)),
                "effective_sample_size": effective_sample_size,
                "effective_sample_size_ratio": ess_ratio,
                "propensity_clipping": clipping,
            },
            balance=balance_rows,
            blocking_issues=blocking_issues,
            warnings=warnings,
            thresholds={
                "extreme_propensity": extreme_propensity_threshold,
                "absolute_smd": balance_threshold,
                "minimum_common_support_fraction": (
                    min_common_support_fraction
                ),
            },
        )
        return _success(result)
    except (ValueError, KeyError, FileNotFoundError) as exc:
        return _error(str(exc))
    except Exception as exc:
        return _error(f"因果诊断执行失败：{exc}")
