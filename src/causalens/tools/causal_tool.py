from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from econml.dml import LinearDML
from econml.inference import StatsModelsInference
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from causalens.schemas import DMLResult
from causalens.tools.causal_readiness import (
    check_causal_readiness,
)


def _success_response(
    result: DMLResult,
) -> dict[str, Any]:
    return {
        "success": True,
        "result": result.model_dump(),
    }


def _error_response(
    message: str,
    **extra: Any,
) -> dict[str, Any]:
    return {
        "success": False,
        "error": message,
        **extra,
    }


def _as_float(
    value: object,
) -> float:
    """将 NumPy 标量或单元素数组转换为 float。"""
    array = np.asarray(value)

    if array.size != 1:
        raise ValueError(
            "Expected a scalar result, "
            f"but received shape {array.shape}."
        )

    return float(
        array.reshape(-1)[0]
    )


def _load_complete_data(
    dataset_path: str,
    treatment: str,
    outcome: str,
    confounders: list[str],
) -> pd.DataFrame:
    try:
        dataframe = pd.read_csv(
            dataset_path
        )
    except FileNotFoundError as exc:
        raise ValueError(
            f"Dataset file does not exist: "
            f"{dataset_path}"
        ) from exc
    except pd.errors.EmptyDataError as exc:
        raise ValueError(
            f"Dataset is empty: "
            f"{dataset_path}"
        ) from exc
    except Exception as exc:
        raise ValueError(
            f"Failed to read dataset: {exc}"
        ) from exc

    required_columns = [
        treatment,
        outcome,
        *confounders,
    ]

    complete_data = (
        dataframe[required_columns]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna()
        .copy()
    )

    if complete_data.empty:
        raise ValueError(
            "No complete observations remain "
            "for DML."
        )

    return complete_data


def run_dml(
    dataset_path: str,
    treatment: str,
    outcome: str,
    confounders: list[str],
    discrete_treatment: bool = True,
    random_state: int = 42,
    cv: int = 5,
) -> dict[str, Any]:
    """
    使用 LinearDML 估计二元处理变量从 0 到 1 的 ATE。

    当前 Week 6 版本：
    - 只支持 0/1 二元 treatment；
    - confounders 作为 W controls；
    - X=None，暂不估计异质性处理效应。
    """
    try:
        normalized_confounders = list(
            confounders or []
        )

        if not discrete_treatment:
            return _error_response(
                "The current Week 6 DML workflow "
                "supports binary discrete treatment "
                "only."
            )

        if (
            not isinstance(cv, int)
            or isinstance(cv, bool)
            or cv < 2
        ):
            return _error_response(
                "cv must be an integer of at least 2."
            )

        # -----------------------------------------
        # 第一层：必须先通过 readiness
        # -----------------------------------------

        readiness_response = (
            check_causal_readiness(
                dataset_path=dataset_path,
                treatment=treatment,
                outcome=outcome,
                confounders=(
                    normalized_confounders
                ),
            )
        )

        if not readiness_response.get(
            "success",
            False,
        ):
            return _error_response(
                "Causal readiness check failed.",
                readiness=readiness_response,
            )

        readiness_result = (
            readiness_response["result"]
        )

        if not readiness_result["ready"]:
            return _error_response(
                "DML was not executed because "
                "causal readiness did not pass.",
                readiness=readiness_result,
            )

        # -----------------------------------------
        # 第二层：确认 treatment 是严格 0/1
        # -----------------------------------------

        complete_data = _load_complete_data(
            dataset_path=dataset_path,
            treatment=treatment,
            outcome=outcome,
            confounders=(
                normalized_confounders
            ),
        )

        observed_treatment_values = sorted(
            complete_data[treatment]
            .drop_duplicates()
            .tolist()
        )

        if observed_treatment_values != [0, 1]:
            return _error_response(
                "Treatment must be coded as 0 and 1 "
                "before running DML. "
                f"Observed values: "
                f"{observed_treatment_values}",
                readiness=readiness_result,
            )

        treatment_counts = (
            complete_data[treatment]
            .value_counts()
            .to_dict()
        )

        smallest_group_size = int(
            min(treatment_counts.values())
        )

        if cv > smallest_group_size:
            return _error_response(
                "cv cannot exceed the size of the "
                "smallest treatment group. "
                f"cv: {cv}; smallest group: "
                f"{smallest_group_size}.",
                readiness=readiness_result,
            )

        # -----------------------------------------
        # 第三层：准备 Y、T、W
        # -----------------------------------------

        y = (
            complete_data[outcome]
            .to_numpy(dtype=float)
        )

        treatment_values = (
            complete_data[treatment]
            .to_numpy(dtype=int)
        )

        controls = (
            complete_data[
                normalized_confounders
            ]
            .astype(float)
        )

        # 结果变量 nuisance model。
        model_y = RandomForestRegressor(
            n_estimators=100,
            min_samples_leaf=10,
            random_state=random_state,
            n_jobs=-1,
        )

        # 二元 treatment nuisance model。
        # scaler 位于 Pipeline 内，避免交叉拟合前泄漏。
        model_t = Pipeline(
            steps=[
                (
                    "scaler",
                    StandardScaler(),
                ),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=2000,
                        random_state=random_state,
                    ),
                ),
            ]
        )

        estimator = LinearDML(
            model_y=model_y,
            model_t=model_t,
            discrete_treatment=True,
            cv=cv,
            random_state=random_state,
        )

        estimator.fit(
            y,
            treatment_values,
            X=None,
            W=controls,
            inference=StatsModelsInference(
                cov_type="HC1"
            ),
        )

        # -----------------------------------------
        # 第四层：提取 ATE 和置信区间
        # -----------------------------------------

        ate = _as_float(
            estimator.ate(
                X=None,
                T0=0,
                T1=1,
            )
        )

        lower_bound, upper_bound = (
            estimator.ate_interval(
                X=None,
                T0=0,
                T1=1,
                alpha=0.05,
            )
        )

        lower_bound = _as_float(
            lower_bound
        )
        upper_bound = _as_float(
            upper_bound
        )

        if lower_bound > upper_bound:
            raise ValueError(
                "The estimated confidence interval "
                "has an invalid order."
            )

        # -----------------------------------------
        # 第五层：原始组均值差，作为对照
        # -----------------------------------------

        treated_mean = float(
            complete_data.loc[
                complete_data[treatment] == 1,
                outcome,
            ].mean()
        )

        control_mean = float(
            complete_data.loc[
                complete_data[treatment] == 0,
                outcome,
            ].mean()
        )

        naive_group_difference = (
            treated_mean - control_mean
        )

        result = DMLResult(
            treatment=treatment,
            outcome=outcome,
            confounders=(
                normalized_confounders
            ),
            sample_size=int(
                len(complete_data)
            ),
            discrete_treatment=True,
            ate=ate,
            confidence_interval=[
                lower_bound,
                upper_bound,
            ],
            confidence_level=0.95,
            naive_group_difference=(
                naive_group_difference
            ),
            model_configuration={
                "estimator": "LinearDML",
                "effect_definition": (
                    "T=1 versus T=0"
                ),
                "model_y": (
                    "RandomForestRegressor"
                ),
                "model_t": (
                    "StandardScaler + "
                    "LogisticRegression"
                ),
                "cv": cv,
                "random_state": (
                    random_state
                ),
                "inference": (
                    "StatsModelsInference"
                ),
                "inference_covariance": (
                    "HC1"
                ),
                "effect_modifiers": None,
                "controls_argument": "W",
            },
            readiness_summary={
                "ready": True,
                "sample_summary": (
                    readiness_result[
                        "sample_summary"
                    ]
                ),
                "thresholds": (
                    readiness_result[
                        "thresholds"
                    ]
                ),
            },
            assumptions=[
                (
                    "No unmeasured confounding after "
                    "conditioning on the supplied "
                    "confounders."
                ),
                (
                    "Treatment positivity or overlap: "
                    "comparable units have a non-zero "
                    "chance of receiving either "
                    "treatment."
                ),
                (
                    "The treatment and outcome are "
                    "consistently defined."
                ),
                (
                    "The supplied nuisance models are "
                    "sufficiently accurate for the "
                    "current analysis."
                ),
            ],
            warnings=[
                *readiness_result["warnings"],
                (
                    "This result should be "
                    "interpreted as causal "
                    "exploration, not definitive "
                    "proof."
                ),
                (
                    "Only the observed confounders "
                    "supplied to the tool were "
                    "adjusted for."
                ),
            ],
        )

        return _success_response(
            result
        )

    except ValueError as exc:
        return _error_response(
            str(exc)
        )
    except Exception as exc:
        return _error_response(
            f"LinearDML analysis failed: {exc}"
        )