from __future__ import annotations

from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd
from econml.dml import CausalForestDML, LinearDML
from econml.inference import StatsModelsInference
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from causalens.causal.evidence_grade import build_causal_evidence_grade
from causalens.causal.data_quality import assess_causal_data_quality
from causalens.causal.design_router import select_causal_design
from causalens.causal.estimators import ESTIMATOR_REGISTRY
from causalens.causal.placebo import run_residualized_treatment_permutation
from causalens.causal.runtime import AnalysisResultCache, build_analysis_cache_key
from causalens.causal.sensitivity import (
    build_observed_confounder_benchmark,
    build_sensitivity_analysis,
)
from causalens.schemas import (
    AIPWResult,
    CausalForestResult,
    CausalEstimationSuiteResult,
    DIDResult,
    DMLResult,
    EventStudyResult,
    IVResult,
    RDDResult,
    StaggeredDIDResult,
    SyntheticControlResult,
    TimeSeriesLocalProjectionResult,
)
from causalens.tools.causal_readiness import (
    check_causal_readiness,
)
from causalens.tools.did_tool import run_difference_in_differences
from causalens.tools.event_study_tool import run_event_study
from causalens.tools.iv_tool import run_two_stage_least_squares
from causalens.tools.rdd_tool import run_sharp_regression_discontinuity
from causalens.tools.staggered_did_tool import run_staggered_did
from causalens.tools.synthetic_control_tool import run_synthetic_control
from causalens.tools.time_series_tool import run_time_series_local_projection


def _success_response(
    result: (
        DMLResult
        | AIPWResult
        | CausalForestResult
        | DIDResult
        | EventStudyResult
        | IVResult
        | RDDResult
        | StaggeredDIDResult
        | SyntheticControlResult
        | TimeSeriesLocalProjectionResult
        | CausalEstimationSuiteResult
    ),
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


def run_aipw(
    dataset_path: str,
    treatment: str,
    outcome: str,
    confounders: list[str],
    random_state: int = 42,
    cv: int = 5,
    propensity_clip: float = 0.01,
) -> dict[str, Any]:
    """使用交叉拟合 AIPW 估计二元处理的 ATE。"""

    try:
        normalized_confounders = list(confounders or [])
        if not isinstance(cv, int) or isinstance(cv, bool) or cv < 2:
            return _error_response("cv must be an integer of at least 2.")
        if not 0 < propensity_clip < 0.5:
            return _error_response(
                "propensity_clip must be between 0 and 0.5."
            )

        readiness_response = check_causal_readiness(
            dataset_path=dataset_path,
            treatment=treatment,
            outcome=outcome,
            confounders=normalized_confounders,
        )
        if not readiness_response.get("success", False):
            return _error_response(
                "Causal readiness check failed.",
                readiness=readiness_response,
            )

        readiness_result = readiness_response["result"]
        if not readiness_result["ready"]:
            return _error_response(
                "AIPW was not executed because causal readiness did not pass.",
                readiness=readiness_result,
            )

        complete_data = _load_complete_data(
            dataset_path=dataset_path,
            treatment=treatment,
            outcome=outcome,
            confounders=normalized_confounders,
        )
        observed_values = sorted(
            complete_data[treatment].drop_duplicates().tolist()
        )
        if observed_values != [0, 1]:
            return _error_response(
                "Treatment must be coded as 0 and 1 before running AIPW. "
                f"Observed values: {observed_values}",
                readiness=readiness_result,
            )

        y = complete_data[outcome].to_numpy(dtype=float)
        treatment_values = complete_data[treatment].to_numpy(dtype=int)
        controls = complete_data[normalized_confounders].to_numpy(
            dtype=float
        )
        treatment_counts = np.bincount(treatment_values, minlength=2)
        if cv > int(treatment_counts.min()):
            return _error_response(
                "cv cannot exceed the size of the smallest treatment group. "
                f"cv: {cv}; smallest group: {int(treatment_counts.min())}.",
                readiness=readiness_result,
            )

        propensity = np.empty(len(complete_data), dtype=float)
        outcome_if_control = np.empty(len(complete_data), dtype=float)
        outcome_if_treated = np.empty(len(complete_data), dtype=float)
        splitter = StratifiedKFold(
            n_splits=cv,
            shuffle=True,
            random_state=random_state,
        )

        for fold_index, (train_index, validation_index) in enumerate(
            splitter.split(controls, treatment_values)
        ):
            fold_seed = random_state + fold_index
            propensity_model = Pipeline(
                steps=[
                    ("scaler", StandardScaler()),
                    (
                        "classifier",
                        LogisticRegression(
                            max_iter=2000,
                            random_state=fold_seed,
                        ),
                    ),
                ]
            )
            propensity_model.fit(
                controls[train_index],
                treatment_values[train_index],
            )
            propensity[validation_index] = propensity_model.predict_proba(
                controls[validation_index]
            )[:, 1]

            for group_value, target in (
                (0, outcome_if_control),
                (1, outcome_if_treated),
            ):
                group_train = train_index[
                    treatment_values[train_index] == group_value
                ]
                outcome_model = RandomForestRegressor(
                    n_estimators=100,
                    min_samples_leaf=10,
                    random_state=fold_seed + group_value,
                    n_jobs=-1,
                )
                outcome_model.fit(
                    controls[group_train],
                    y[group_train],
                )
                target[validation_index] = outcome_model.predict(
                    controls[validation_index]
                )

        clipped_propensity = np.clip(
            propensity,
            propensity_clip,
            1.0 - propensity_clip,
        )
        influence_scores = (
            outcome_if_treated
            - outcome_if_control
            + treatment_values
            * (y - outcome_if_treated)
            / clipped_propensity
            - (1 - treatment_values)
            * (y - outcome_if_control)
            / (1.0 - clipped_propensity)
        )
        ate = float(np.mean(influence_scores))
        standard_error = float(
            np.std(influence_scores, ddof=1) / np.sqrt(len(influence_scores))
        )
        critical_value = NormalDist().inv_cdf(0.975)
        confidence_interval = [
            float(ate - critical_value * standard_error),
            float(ate + critical_value * standard_error),
        ]
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
        clipped_fraction = float(
            np.mean(
                (propensity < propensity_clip)
                | (propensity > 1.0 - propensity_clip)
            )
        )
        warnings = list(readiness_result["warnings"])
        if clipped_fraction > 0:
            warnings.append(
                "Some out-of-fold propensity scores were clipped for AIPW."
            )
        warnings.extend(
            [
                (
                    "This result should be interpreted as causal exploration, "
                    "not definitive proof."
                ),
                (
                    "AIPW is doubly robust to one nuisance-model misspecification, "
                    "not to unmeasured confounding or simultaneous misspecification."
                ),
            ]
        )

        result = AIPWResult(
            treatment=treatment,
            outcome=outcome,
            confounders=normalized_confounders,
            sample_size=int(len(complete_data)),
            ate=ate,
            standard_error=standard_error,
            confidence_interval=confidence_interval,
            naive_group_difference=treated_mean - control_mean,
            model_configuration={
                "estimator": "AIPW",
                "effect_definition": "T=1 versus T=0",
                "cross_fitting_folds": cv,
                "propensity_model": (
                    "StandardScaler + LogisticRegression"
                ),
                "outcome_models": (
                    "group-specific RandomForestRegressor"
                ),
                "propensity_clip": propensity_clip,
                "clipped_fraction": clipped_fraction,
                "random_state": random_state,
            },
            readiness_summary={
                "ready": True,
                "sample_summary": readiness_result["sample_summary"],
                "thresholds": readiness_result["thresholds"],
            },
            assumptions=[
                (
                    "No unmeasured confounding after conditioning on the supplied "
                    "confounders."
                ),
                "Treatment positivity and consistent treatment definition.",
                (
                    "At least one of the propensity or outcome nuisance models is "
                    "correctly specified enough for the doubly robust guarantee."
                ),
            ],
            warnings=warnings,
        )
        return _success_response(result)
    except ValueError as exc:
        return _error_response(str(exc))
    except Exception as exc:
        return _error_response(f"AIPW analysis failed: {exc}")


def run_causal_forest(
    dataset_path: str,
    treatment: str,
    outcome: str,
    confounders: list[str],
    effect_modifiers: list[str],
    random_state: int = 42,
    cv: int = 5,
) -> dict[str, Any]:
    """使用 CausalForestDML 估计 CATE 及其样本异质性。"""

    try:
        normalized_confounders = list(dict.fromkeys(confounders or []))
        normalized_modifiers = list(dict.fromkeys(effect_modifiers or []))
        if not normalized_modifiers:
            return _error_response(
                "CATE estimation requires at least one effect modifier."
            )
        if not isinstance(cv, int) or isinstance(cv, bool) or cv < 2:
            return _error_response("cv must be an integer of at least 2.")
        forbidden = {treatment, outcome} & set(normalized_modifiers)
        if forbidden:
            return _error_response(
                "Treatment or outcome cannot be used as an effect modifier: "
                f"{sorted(forbidden)}"
            )

        readiness_response = check_causal_readiness(
            dataset_path=dataset_path,
            treatment=treatment,
            outcome=outcome,
            confounders=normalized_confounders,
        )
        if not readiness_response.get("success", False):
            return _error_response(
                "Causal readiness check failed.",
                readiness=readiness_response,
            )
        readiness_result = readiness_response["result"]
        if not readiness_result["ready"]:
            return _error_response(
                "CausalForestDML was not executed because causal readiness "
                "did not pass.",
                readiness=readiness_result,
            )

        analysis_columns = list(
            dict.fromkeys([*normalized_confounders, *normalized_modifiers])
        )
        complete_data = _load_complete_data(
            dataset_path=dataset_path,
            treatment=treatment,
            outcome=outcome,
            confounders=analysis_columns,
        )
        observed_values = sorted(
            complete_data[treatment].drop_duplicates().tolist()
        )
        if observed_values != [0, 1]:
            return _error_response(
                "Treatment must be coded as 0 and 1 before running "
                "CausalForestDML."
            )

        y = complete_data[outcome].to_numpy(dtype=float)
        treatment_values = complete_data[treatment].to_numpy(dtype=int)
        modifiers = complete_data[normalized_modifiers].to_numpy(dtype=float)
        control_columns = [
            column
            for column in normalized_confounders
            if column not in set(normalized_modifiers)
        ]
        controls = (
            complete_data[control_columns].to_numpy(dtype=float)
            if control_columns
            else None
        )
        treatment_counts = np.bincount(treatment_values, minlength=2)
        if cv > int(treatment_counts.min()):
            return _error_response(
                "cv cannot exceed the size of the smallest treatment group."
            )

        model_y = RandomForestRegressor(
            n_estimators=100,
            min_samples_leaf=10,
            random_state=random_state,
            n_jobs=-1,
        )
        model_t = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=2000,
                        random_state=random_state,
                    ),
                ),
            ]
        )
        estimator = CausalForestDML(
            model_y=model_y,
            model_t=model_t,
            discrete_treatment=True,
            cv=cv,
            n_estimators=200,
            min_samples_leaf=10,
            honest=True,
            inference=True,
            random_state=random_state,
            n_jobs=-1,
        )
        estimator.fit(
            y,
            treatment_values,
            X=modifiers,
            W=controls,
            inference="auto",
        )

        individual_effects = np.asarray(
            estimator.effect(modifiers, T0=0, T1=1),
            dtype=float,
        ).reshape(-1)
        average_cate = float(np.mean(individual_effects))
        lower, upper = estimator.ate_interval(
            modifiers,
            T0=0,
            T1=1,
            alpha=0.05,
        )
        average_interval = [_as_float(lower), _as_float(upper)]
        standard_error = float(
            (average_interval[1] - average_interval[0])
            / (2.0 * NormalDist().inv_cdf(0.975))
        )

        subgroup_summaries: list[dict[str, Any]] = []
        for index, modifier_name in enumerate(normalized_modifiers):
            modifier_values = modifiers[:, index]
            median = float(np.median(modifier_values))
            lower_group = individual_effects[modifier_values <= median]
            upper_group = individual_effects[modifier_values > median]
            subgroup_summaries.append(
                {
                    "effect_modifier": modifier_name,
                    "split_value": median,
                    "lower_or_equal_mean_cate": float(np.mean(lower_group)),
                    "upper_mean_cate": float(np.mean(upper_group)),
                    "upper_minus_lower_cate": float(
                        np.mean(upper_group) - np.mean(lower_group)
                    ),
                }
            )

        importances = np.asarray(
            estimator.feature_importances_,
            dtype=float,
        ).reshape(-1)
        feature_importances = {
            name: float(value)
            for name, value in zip(normalized_modifiers, importances)
        }
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
        result = CausalForestResult(
            treatment=treatment,
            outcome=outcome,
            confounders=normalized_confounders,
            effect_modifiers=normalized_modifiers,
            sample_size=int(len(complete_data)),
            ate=average_cate,
            standard_error=standard_error,
            confidence_interval=average_interval,
            naive_group_difference=treated_mean - control_mean,
            heterogeneity={
                "cate_mean": average_cate,
                "cate_standard_deviation": float(
                    np.std(individual_effects, ddof=1)
                ),
                "cate_minimum": float(np.min(individual_effects)),
                "cate_p10": float(np.quantile(individual_effects, 0.10)),
                "cate_p25": float(np.quantile(individual_effects, 0.25)),
                "cate_median": float(np.median(individual_effects)),
                "cate_p75": float(np.quantile(individual_effects, 0.75)),
                "cate_p90": float(np.quantile(individual_effects, 0.90)),
                "cate_maximum": float(np.max(individual_effects)),
                "positive_effect_fraction": float(
                    np.mean(individual_effects > 0)
                ),
                "feature_importances": feature_importances,
                "median_split_subgroups": subgroup_summaries,
            },
            model_configuration={
                "estimator": "CausalForestDML",
                "effect_definition": "T=1 versus T=0",
                "effect_modifiers_argument": "X",
                "controls_argument": "W",
                "n_estimators": 200,
                "min_samples_leaf": 10,
                "honest": True,
                "cross_fitting_folds": cv,
                "random_state": random_state,
            },
            readiness_summary={
                "ready": True,
                "sample_summary": readiness_result["sample_summary"],
                "thresholds": readiness_result["thresholds"],
            },
            assumptions=[
                (
                    "No unmeasured confounding after conditioning on supplied "
                    "confounders and effect modifiers."
                ),
                "Treatment positivity and consistent treatment definition.",
                (
                    "The selected effect modifiers are pre-treatment variables "
                    "and the forest has enough support to estimate heterogeneity."
                ),
            ],
            warnings=[
                *readiness_result["warnings"],
                (
                    "CATE patterns are exploratory and require external or "
                    "held-out validation before subgroup decisions."
                ),
                (
                    "Median-split subgroup summaries are descriptive views of "
                    "forest predictions, not separate randomized comparisons."
                ),
            ],
        )
        return _success_response(result)
    except (ValueError, KeyError) as exc:
        return _error_response(str(exc))
    except Exception as exc:
        return _error_response(f"CausalForestDML analysis failed: {exc}")


def _normalize_estimate(
    estimator_name: str,
    result: dict[str, Any],
) -> dict[str, Any]:
    return {
        "estimator": estimator_name,
        "analysis_type": result.get("analysis_type"),
        "ate": result.get("ate"),
        "standard_error": result.get("standard_error"),
        "confidence_interval": result.get("confidence_interval"),
        "confidence_level": result.get("confidence_level", 0.95),
        "sample_size": result.get("sample_size"),
        "naive_group_difference": result.get("naive_group_difference"),
        "unadjusted_did": result.get("unadjusted_did"),
        "cell_means": result.get("cell_means") or {},
        "model_configuration": result.get("model_configuration") or {},
        "assumptions": result.get("assumptions") or [],
        "warnings": result.get("warnings") or [],
        "heterogeneity": result.get("heterogeneity") or {},
        "effect_modifiers": result.get("effect_modifiers") or [],
        "dynamic_effects": result.get("dynamic_effects") or [],
        "group_time_effects": result.get("group_time_effects") or [],
        "pretrend_test": result.get("pretrend_test") or {},
        "placebo_test": result.get("placebo_test") or {},
        "cohort_sensitivity": result.get("cohort_sensitivity") or {},
        "first_stage": result.get("first_stage") or {},
        "endogeneity_test": result.get("endogeneity_test") or {},
        "overidentification_test": (
            result.get("overidentification_test") or {}
        ),
        "local_support": result.get("local_support") or {},
        "density_diagnostic": result.get("density_diagnostic") or {},
        "covariate_continuity": result.get("covariate_continuity") or {},
        "bandwidth_sensitivity": result.get("bandwidth_sensitivity") or {},
        "donut_sensitivity": result.get("donut_sensitivity") or {},
        "running_variable": result.get("running_variable"),
        "cutoff": result.get("cutoff"),
        "bandwidth": result.get("bandwidth"),
        "treatment_above_cutoff": result.get("treatment_above_cutoff"),
        "intervention_time": result.get("intervention_time"),
        "treated_unit": result.get("treated_unit"),
        "donor_weights": result.get("donor_weights") or [],
        "trajectories": result.get("trajectories") or [],
        "fit_diagnostics": result.get("fit_diagnostics") or {},
        "placebo_inference": result.get("placebo_inference") or {},
        "stationarity": result.get("stationarity") or {},
        "residual_diagnostics": result.get("residual_diagnostics") or {},
        "chart_path": result.get("chart_path"),
    }


def run_causal_estimators(
    dataset_path: str,
    treatment: str,
    outcome: str,
    confounders: list[str],
    effect_modifiers: list[str] | None = None,
    negative_control_outcomes: list[str] | None = None,
    instruments: list[str] | None = None,
    estimand: str = "ATE",
    design_type: str = "cross_sectional",
    post_column: str | None = None,
    event_time_column: str | None = None,
    reference_period: int = -1,
    time_column: str | None = None,
    treatment_start_column: str | None = None,
    unit_id_column: str | None = None,
    running_variable: str | None = None,
    cutoff: float | None = None,
    treatment_above_cutoff: bool = True,
    bandwidth: float | None = None,
    intervention_time: float | None = None,
    treated_unit: str | int | float | None = None,
    max_horizon: int = 5,
    treatment_lags: int = 1,
    outcome_lags: int = 1,
    control_lag: int = 1,
    cumulative_outcome: bool = True,
    hac_lags: int | None = None,
    chart_output_dir: str | None = None,
    bootstrap_iterations: int = 200,
    max_propensity_clip_fraction: float = 0.10,
    min_weighted_control_ess: float = 10.0,
    permutation_iterations: int = 200,
    random_state: int = 42,
    cv: int = 5,
    result_cache_dir: str | None = None,
) -> dict[str, Any]:
    """按注册表运行适配 estimand 的主估计器及可用稳健性估计器。"""

    normalized_negative_controls = list(
        dict.fromkeys(negative_control_outcomes or [])
    )
    normalized_instruments = list(dict.fromkeys(instruments or []))
    cache: AnalysisResultCache | None = None
    cache_key: str | None = None
    if result_cache_dir:
        cache_request = {
            "treatment": treatment,
            "outcome": outcome,
            "confounders": list(confounders),
            "effect_modifiers": list(effect_modifiers or []),
            "negative_control_outcomes": normalized_negative_controls,
            "instruments": normalized_instruments,
            "estimand": estimand,
            "design_type": design_type,
            "post_column": post_column,
            "event_time_column": event_time_column,
            "reference_period": reference_period,
            "time_column": time_column,
            "treatment_start_column": treatment_start_column,
            "unit_id_column": unit_id_column,
            "running_variable": running_variable,
            "cutoff": cutoff,
            "treatment_above_cutoff": treatment_above_cutoff,
            "bandwidth": bandwidth,
            "intervention_time": intervention_time,
            "treated_unit": treated_unit,
            "max_horizon": max_horizon,
            "treatment_lags": treatment_lags,
            "outcome_lags": outcome_lags,
            "control_lag": control_lag,
            "cumulative_outcome": cumulative_outcome,
            "hac_lags": hac_lags,
            "bootstrap_iterations": bootstrap_iterations,
            "permutation_iterations": permutation_iterations,
            "random_state": random_state,
            "cv": cv,
        }
        try:
            cache = AnalysisResultCache(result_cache_dir)
            cache_key = build_analysis_cache_key(
                dataset_path=dataset_path,
                request=cache_request,
            )
            cached = cache.load(cache_key)
            if cached and cached.get("success") is True:
                cached_result = cached.get("result") or {}
                if isinstance(cached_result, dict):
                    cached_result["runtime"] = {
                        **(cached_result.get("runtime") or {}),
                        "cache_enabled": True,
                        "cache_hit": True,
                        "cache_key": cache_key,
                    }
                return cached
        except (OSError, ValueError):
            cache = None
            cache_key = None
    if len(normalized_instruments) != len(instruments or []):
        return _error_response("instruments contains duplicates.")
    forbidden_instruments = {treatment, outcome, *confounders} & set(
        normalized_instruments
    )
    if forbidden_instruments:
        return _error_response(
            "Instruments must be distinct from treatment, outcome, and "
            f"confounders: {sorted(forbidden_instruments)}"
        )
    if len(normalized_negative_controls) != len(negative_control_outcomes or []):
        return _error_response("negative_control_outcomes contains duplicates.")
    forbidden_negative_controls = {treatment, outcome, *confounders} & set(
        normalized_negative_controls
    )
    if forbidden_negative_controls:
        return _error_response(
            "Negative-control outcomes must be distinct from treatment, the "
            "primary outcome, and confounders: "
            f"{sorted(forbidden_negative_controls)}"
        )

    design_selection = select_causal_design(
        design_type=design_type,
        estimand=estimand,
        post_column=post_column,
        event_time_column=event_time_column,
        time_column=time_column,
        treatment_start_column=treatment_start_column,
        unit_id_column=unit_id_column,
        running_variable=running_variable,
        cutoff=cutoff,
        intervention_time=intervention_time,
        instruments=normalized_instruments,
        effect_modifiers=list(effect_modifiers or []),
    )
    if design_selection["ready"] is not True:
        return _error_response(
            "No registered estimator supports the requested causal design: "
            + "; ".join(design_selection["blocking_issues"]),
            design_selection=design_selection,
        )
    selected = [
        ESTIMATOR_REGISTRY[key]
        for key in design_selection["selected_estimators"]
    ]

    design_columns = [
        column
        for column in (
            post_column,
            event_time_column,
            time_column,
            treatment_start_column,
            unit_id_column,
            running_variable,
        )
        if column
    ]
    required_columns = list(
        dict.fromkeys(
            [
                treatment,
                outcome,
                *confounders,
                *normalized_instruments,
                *design_columns,
            ]
        )
    )
    non_numeric_roles = {unit_id_column} if unit_id_column else set()
    if design_type == "time_series" and time_column:
        non_numeric_roles.add(time_column)
    numeric_columns = [
        column for column in required_columns if column not in non_numeric_roles
    ]
    data_quality_response = assess_causal_data_quality(
        dataset_path=dataset_path,
        required_columns=required_columns,
        numeric_columns=numeric_columns,
        adjustment_columns=list(confounders),
        panel_key=(unit_id_column, time_column)
        if unit_id_column and time_column
        else None,
    )
    if data_quality_response.get("success") is not True:
        return _error_response(
            str(data_quality_response.get("error") or "Data-quality check failed."),
            design_selection=design_selection,
        )
    data_quality = data_quality_response["result"]
    if data_quality["ready"] is not True:
        return _error_response(
            "Causal data-quality gate did not pass: "
            + "; ".join(data_quality["blocking_issues"]),
            data_quality=data_quality,
            design_selection=design_selection,
        )

    runners = {
        "aipw": run_aipw,
        "linear_dml": run_dml,
        "causal_forest_dml": run_causal_forest,
        "difference_in_differences": run_difference_in_differences,
        "event_study": run_event_study,
        "staggered_adoption_did": run_staggered_did,
        "two_stage_least_squares": run_two_stage_least_squares,
        "sharp_regression_discontinuity": (
            run_sharp_regression_discontinuity
        ),
        "synthetic_control": run_synthetic_control,
        "time_series_local_projection": run_time_series_local_projection,
    }
    estimates: list[dict[str, Any]] = []
    failed_estimators: list[dict[str, str]] = []
    primary_runner_arguments: dict[str, Any] | None = None
    primary_runner_key: str | None = None
    primary_display_name: str | None = None
    for spec in selected:
        runner_arguments: dict[str, Any] = {
            "dataset_path": dataset_path,
            "treatment": treatment,
            "outcome": outcome,
            "confounders": confounders,
            "random_state": random_state,
            "cv": cv,
        }
        if spec.key == "causal_forest_dml":
            runner_arguments["effect_modifiers"] = list(
                effect_modifiers or []
            )
        if spec.key == "difference_in_differences":
            if not post_column:
                return _error_response(
                    "Difference-in-Differences requires post_column."
                )
            runner_arguments = {
                "dataset_path": dataset_path,
                "group_column": treatment,
                "post_column": post_column,
                "outcome": outcome,
                "confounders": confounders,
                "unit_id_column": unit_id_column,
            }
        if spec.key == "event_study":
            if not event_time_column:
                return _error_response("Event Study requires event_time_column.")
            runner_arguments = {
                "dataset_path": dataset_path,
                "group_column": treatment,
                "event_time_column": event_time_column,
                "outcome": outcome,
                "confounders": confounders,
                "unit_id_column": unit_id_column,
                "post_column": post_column,
                "reference_period": reference_period,
                "chart_output_dir": chart_output_dir,
            }
        if spec.key == "staggered_adoption_did":
            if not time_column or not treatment_start_column or not unit_id_column:
                return _error_response(
                    "Staggered-adoption DiD requires time_column, "
                    "treatment_start_column, and unit_id_column."
                )
            runner_arguments = {
                "dataset_path": dataset_path,
                "treatment_column": treatment,
                "time_column": time_column,
                "treatment_start_column": treatment_start_column,
                "outcome": outcome,
                "unit_id_column": unit_id_column,
                "confounders": confounders,
                "random_state": random_state,
                "bootstrap_iterations": bootstrap_iterations,
                "cross_fitting_folds": cv,
                "max_propensity_clip_fraction": max_propensity_clip_fraction,
                "min_weighted_control_ess": min_weighted_control_ess,
                "chart_output_dir": chart_output_dir,
            }
        if spec.key == "two_stage_least_squares":
            runner_arguments = {
                "dataset_path": dataset_path,
                "treatment": treatment,
                "outcome": outcome,
                "instruments": normalized_instruments,
                "confounders": confounders,
            }
        if spec.key == "sharp_regression_discontinuity":
            runner_arguments = {
                "dataset_path": dataset_path,
                "treatment": treatment,
                "outcome": outcome,
                "running_variable": running_variable,
                "cutoff": cutoff,
                "treatment_above_cutoff": treatment_above_cutoff,
                "covariates": confounders,
                "bandwidth": bandwidth,
            }
        if spec.key == "synthetic_control":
            if not unit_id_column or not time_column or intervention_time is None:
                return _error_response(
                    "Synthetic control requires unit_id_column, time_column, and "
                    "intervention_time."
                )
            runner_arguments = {
                "dataset_path": dataset_path,
                "treatment": treatment,
                "outcome": outcome,
                "unit_id_column": unit_id_column,
                "time_column": time_column,
                "intervention_time": intervention_time,
                "covariates": confounders,
                "treated_unit": treated_unit,
            }
        if spec.key == "time_series_local_projection":
            if not time_column:
                return _error_response(
                    "Time-series local projection requires time_column."
                )
            runner_arguments = {
                "dataset_path": dataset_path,
                "treatment": treatment,
                "outcome": outcome,
                "time_column": time_column,
                "confounders": confounders,
                "max_horizon": max_horizon,
                "treatment_lags": treatment_lags,
                "outcome_lags": outcome_lags,
                "control_lag": control_lag,
                "cumulative_outcome": cumulative_outcome,
                "hac_lags": hac_lags,
            }
        if spec.role == "primary":
            primary_runner_arguments = dict(runner_arguments)
            primary_runner_key = spec.key
            primary_display_name = spec.display_name
        response = runners[spec.key](
            **runner_arguments
        )
        if response.get("success") is True:
            estimates.append(
                _normalize_estimate(spec.display_name, response["result"])
            )
        else:
            failed_estimators.append(
                {
                    "estimator": spec.display_name,
                    "error": str(response.get("error") or "Unknown error"),
                }
            )
            if spec.role == "primary":
                return _error_response(
                    f"Primary estimator {spec.display_name} failed: "
                    f"{failed_estimators[-1]['error']}",
                    failed_estimators=failed_estimators,
                )

    negative_control_results: list[dict[str, Any]] = []
    if normalized_negative_controls:
        if not primary_runner_arguments or not primary_runner_key:
            return _error_response(
                "Primary estimator arguments were unavailable for negative controls."
            )
        for negative_outcome in normalized_negative_controls:
            placebo_arguments = dict(primary_runner_arguments)
            placebo_arguments["outcome"] = negative_outcome
            response = runners[primary_runner_key](**placebo_arguments)
            if response.get("success") is not True:
                negative_control_results.append(
                    {
                        "outcome": negative_outcome,
                        "estimator": primary_display_name,
                        "success": False,
                        "passed": False,
                        "error": str(response.get("error") or "Unknown error"),
                    }
                )
                continue
            placebo = response["result"]
            interval = [
                float(placebo["confidence_interval"][0]),
                float(placebo["confidence_interval"][1]),
            ]
            negative_control_results.append(
                {
                    "outcome": negative_outcome,
                    "estimator": primary_display_name,
                    "success": True,
                    "passed": bool(interval[0] <= 0 <= interval[1]),
                    "estimate": float(placebo["ate"]),
                    "standard_error": (
                        float(placebo["standard_error"])
                        if placebo.get("standard_error") is not None
                        else None
                    ),
                    "confidence_interval": interval,
                    "interpretation": (
                        "Pass means the negative-control outcome interval includes "
                        "zero; failure is a falsification warning, not proof of bias."
                    ),
                }
            )

    primary = estimates[0]
    for estimate in estimates:
        if design_type in {
            "instrumental_variable",
            "regression_discontinuity",
            "synthetic_control",
            "time_series",
        }:
            estimate["sensitivity_analysis"] = {
                "available": False,
                "method": (
                    "not_applicable_to_iv_2sls"
                    if design_type == "instrumental_variable"
                    else "not_applicable_to_local_linear_rdd"
                    if design_type == "regression_discontinuity"
                    else "not_applicable_to_synthetic_control"
                    if design_type == "synthetic_control"
                    else "not_applicable_to_time_series_local_projection"
                ),
                "reason": (
                    "The regression partial-R² sensitivity approximation used "
                    "for adjustment estimators is not applied to this design."
                ),
                "unmeasured_confounding_risk": "not_applicable",
            }
            continue
        try:
            estimate["sensitivity_analysis"] = build_sensitivity_analysis(
                estimate=float(estimate["ate"]),
                standard_error=(
                    float(estimate["standard_error"])
                    if estimate.get("standard_error") is not None
                    else None
                ),
                confidence_interval=estimate.get("confidence_interval"),
                sample_size=int(estimate["sample_size"]),
                adjusted_covariate_count=len(confounders),
                confidence_level=float(estimate.get("confidence_level", 0.95)),
            )
        except ValueError as exc:
            estimate["sensitivity_analysis"] = {
                "available": False,
                "method": "cinelli_hazlett_partial_r2",
                "reason": str(exc),
                "unmeasured_confounding_risk": "unavailable",
            }
    try:
        benchmark_frame = pd.read_csv(dataset_path)
        for estimate in estimates:
            sensitivity = estimate["sensitivity_analysis"]
            if not sensitivity.get("available"):
                continue
            sensitivity["observed_confounder_benchmark"] = (
                build_observed_confounder_benchmark(
                    dataframe=benchmark_frame,
                    treatment=treatment,
                    outcome=outcome,
                    confounders=list(confounders),
                    robustness_value_point_estimate=float(
                        sensitivity["robustness_value_point_estimate"]
                    ),
                    robustness_value_significance=float(
                        sensitivity["robustness_value_significance"]
                    ),
                )
            )
    except Exception as exc:
        for estimate in estimates:
            sensitivity = estimate["sensitivity_analysis"]
            if sensitivity.get("available"):
                sensitivity["observed_confounder_benchmark"] = {
                    "available": False,
                    "reason": f"Observed-confounder benchmarking failed: {exc}",
                    "confounders": [],
                }
    primary_sensitivity = primary["sensitivity_analysis"]
    primary_benchmark = primary_sensitivity.get(
        "observed_confounder_benchmark", {}
    )
    if design_type == "cross_sectional":
        try:
            permutation_placebo = run_residualized_treatment_permutation(
                dataframe=pd.read_csv(dataset_path),
                treatment=treatment,
                outcome=outcome,
                confounders=list(
                    dict.fromkeys([*confounders, *(effect_modifiers or [])])
                ),
                iterations=permutation_iterations,
                random_state=random_state,
            )
        except Exception as exc:
            permutation_placebo = {
                "available": False,
                "method": "residualized_treatment_permutation",
                "reason": str(exc),
            }
    else:
        permutation_placebo = {
            "available": False,
            "method": "residualized_treatment_permutation",
            "reason": (
                "Residualized-treatment permutation is only applied to "
                "cross-sectional adjustment estimators. DiD uses timing and "
                "pretrend diagnostics; IV uses first-stage and instrument "
                "diagnostics; RDD and synthetic control use design-specific "
                "local or placebo diagnostics."
            ),
        }
    comparison: dict[str, Any] = {
        "estimate_count": len(estimates),
        "failed_estimators": failed_estimators,
    }
    if len(estimates) >= 2:
        ate_values = [float(item["ate"]) for item in estimates]
        intervals = [item["confidence_interval"] for item in estimates]
        comparison.update(
            {
                "ate_min": min(ate_values),
                "ate_max": max(ate_values),
                "absolute_ate_difference": max(ate_values) - min(ate_values),
                "confidence_intervals_overlap": (
                    max(float(interval[0]) for interval in intervals)
                    <= min(float(interval[1]) for interval in intervals)
                ),
            }
        )

    warnings = [
        warning
        for estimate in estimates
        for warning in estimate["warnings"]
    ]
    warnings.extend(data_quality.get("warnings") or [])
    if failed_estimators:
        warnings.append(
            "At least one robustness estimator failed; comparison is incomplete."
        )
    failed_negative_controls = [
        row for row in negative_control_results if row.get("passed") is not True
    ]
    if failed_negative_controls:
        warnings.append(
            "At least one negative-control outcome failed or produced a non-zero "
            "effect interval; inspect the falsification results before causal "
            "interpretation."
        )
    if permutation_placebo.get("available"):
        if permutation_placebo.get("calibration_passed") is not True:
            warnings.append(
                "The permuted-treatment distribution was not adequately centered "
                "around zero under the configured Monte Carlo budget."
            )
        if permutation_placebo.get("signal_status") == (
            "not_distinguishable_from_permuted_labels"
        ):
            warnings.append(
                "The adjusted treatment signal was not distinguishable from the "
                "residualized pseudo-treatment distribution."
            )
    primary_risk = str(primary_sensitivity["unmeasured_confounding_risk"])
    if primary_risk in {"high", "elevated"}:
        warnings.append(
            "The primary estimate is sensitive to relatively weak hypothetical "
            "unmeasured confounding under the partial-R² approximation."
        )
    elif primary_risk == "statistically_inconclusive":
        warnings.append(
            "The primary estimate is already statistically inconclusive at the "
            "reported confidence level."
        )
    elif primary_risk == "unavailable":
        warnings.append(
            "Unmeasured-confounding sensitivity could not be computed for the "
            "primary estimate."
        )
    if primary_benchmark.get("assessment") == (
        "observed_strength_can_explain_away_point_estimate"
    ):
        warnings.append(
            "An omitted confounder with the benchmarked strength of the strongest "
            "observed confounder could explain away the primary point estimate."
        )
    elif primary_benchmark.get("assessment") == (
        "observed_strength_can_remove_significance"
    ):
        warnings.append(
            "An omitted confounder with the benchmarked strength of the strongest "
            "observed confounder could remove primary-estimate significance."
        )
    assumptions = list(
        dict.fromkeys(
            assumption
            for estimate in estimates
            for assumption in estimate["assumptions"]
        )
    )
    negative_control_summary = {
        "requested_outcomes": normalized_negative_controls,
        "estimator": primary_display_name,
        "result_count": len(negative_control_results),
        "passed_count": sum(
            row.get("passed") is True for row in negative_control_results
        ),
        "all_passed": (
            not failed_negative_controls if negative_control_results else None
        ),
        "results": negative_control_results,
        "assumption": (
            "Each declared negative-control outcome is known not to be "
            "caused by the treatment over the study horizon."
        ),
        "multiplicity_note": (
            "Intervals are reported per negative-control outcome without "
            "multiple-testing adjustment."
        ),
    }
    evidence_grade = build_causal_evidence_grade(
        primary_estimate=primary,
        comparison=comparison,
        negative_control_summary=negative_control_summary,
        permutation_placebo=permutation_placebo,
    )
    if evidence_grade["grade"] in {"exploratory_only", "limited_support"}:
        warnings.append(
            "The deterministic causal-evidence grade is "
            f"{evidence_grade['grade']}; inspect its red flags and missing "
            "checks before causal interpretation."
        )
    result = CausalEstimationSuiteResult(
        estimand=estimand,
        design_type=design_type,
        primary_estimator=str(primary["estimator"]),
        robustness_estimators=[
            str(item["estimator"]) for item in estimates[1:]
        ],
        treatment=treatment,
        outcome=outcome,
        confounders=list(confounders),
        instruments=normalized_instruments,
        running_variable=running_variable,
        cutoff=cutoff,
        treatment_above_cutoff=treatment_above_cutoff,
        bandwidth=(
            float(primary["bandwidth"])
            if primary.get("bandwidth") is not None
            else bandwidth
        ),
        intervention_time=intervention_time,
        treated_unit=(
            str(primary["treated_unit"])
            if primary.get("treated_unit") is not None
            else str(treated_unit)
            if treated_unit is not None
            else None
        ),
        sample_size=int(primary["sample_size"]),
        estimates=estimates,
        comparison=comparison,
        sensitivity_summary={
            "method": primary_sensitivity["method"],
            "primary_estimator": primary["estimator"],
            "primary_unmeasured_confounding_risk": primary_risk,
            "primary_robustness_value_point_estimate": primary_sensitivity[
                "robustness_value_point_estimate"
            ] if primary_sensitivity.get("available") else None,
            "primary_robustness_value_significance": primary_sensitivity[
                "robustness_value_significance"
            ] if primary_sensitivity.get("available") else None,
            "available": bool(primary_sensitivity.get("available")),
            "observed_confounder_benchmark": {
                "available": bool(primary_benchmark.get("available")),
                "strongest_confounder": primary_benchmark.get(
                    "strongest_confounder"
                ),
                "strongest_equivalent_robustness_value": primary_benchmark.get(
                    "strongest_equivalent_robustness_value"
                ),
                "assessment": primary_benchmark.get("assessment"),
            },
            "e_value_computed": False,
            "interpretation": (
                primary_sensitivity.get("reason")
                if design_type in {
                    "instrumental_variable",
                    "regression_discontinuity",
                    "synthetic_control",
                    "time_series",
                }
                else "Each robustness value is the equal partial-R² strength an "
                "omitted confounder would need with treatment and outcome to "
                "overturn the corresponding conclusion."
            ),
        },
        negative_control_summary=negative_control_summary,
        permutation_placebo=permutation_placebo,
        evidence_grade=evidence_grade,
        data_quality=data_quality,
        design_selection=design_selection,
        runtime={
            "cache_enabled": cache is not None,
            "cache_hit": False,
            "cache_key": cache_key,
        },
        assumptions=assumptions,
        warnings=list(dict.fromkeys(warnings)),
    )
    response = _success_response(result)
    if cache is not None and cache_key is not None:
        try:
            cache.store(cache_key, response)
        except OSError:
            pass
    return response
