from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from causalens.schemas import CausalReadinessResult


DEFAULT_MIN_COMPLETE_CASES = 100
DEFAULT_MIN_GROUP_SIZE = 30
DEFAULT_MAX_MISSING_RATE = 0.20


def _success_response(
    result: CausalReadinessResult,
) -> dict[str, Any]:
    return {
        "success": True,
        "result": result.model_dump(),
    }


def _error_response(
    message: str,
) -> dict[str, Any]:
    return {
        "success": False,
        "error": message,
    }


def _load_dataset(
    dataset_path: str,
) -> pd.DataFrame:
    if not dataset_path or not dataset_path.strip():
        raise ValueError(
            "dataset_path cannot be empty."
        )

    try:
        dataframe = pd.read_csv(dataset_path)
    except FileNotFoundError as exc:
        raise ValueError(
            f"Dataset file does not exist: "
            f"{dataset_path}"
        ) from exc
    except pd.errors.EmptyDataError as exc:
        raise ValueError(
            f"Dataset is empty: {dataset_path}"
        ) from exc
    except Exception as exc:
        raise ValueError(
            f"Failed to read dataset: {exc}"
        ) from exc

    if dataframe.empty:
        raise ValueError(
            "Dataset contains no rows."
        )

    return dataframe


def _find_duplicates(
    values: list[str],
) -> list[str]:
    counts = Counter(values)

    return sorted(
        value
        for value, count in counts.items()
        if count > 1
    )


def _build_result(
    *,
    ready: bool,
    treatment: str,
    outcome: str,
    confounders: list[str],
    blocking_issues: list[str],
    warnings: list[str],
    sample_summary: dict[str, Any],
    variable_summary: dict[str, Any],
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    result = CausalReadinessResult(
        ready=ready,
        treatment=treatment,
        outcome=outcome,
        confounders=confounders,
        blocking_issues=blocking_issues,
        warnings=warnings,
        sample_summary=sample_summary,
        variable_summary=variable_summary,
        thresholds=thresholds,
    )

    return _success_response(result)


def check_causal_readiness(
    dataset_path: str,
    treatment: str,
    outcome: str,
    confounders: list[str] | None,
    min_complete_cases: int = (
        DEFAULT_MIN_COMPLETE_CASES
    ),
    min_group_size: int = (
        DEFAULT_MIN_GROUP_SIZE
    ),
    max_missing_rate: float = (
        DEFAULT_MAX_MISSING_RATE
    ),
) -> dict[str, Any]:
    """
    检查当前变量配置和 CSV 数据是否满足二元处理
    LinearDML 的最低运行条件。

    该函数只检查技术和数据层面的前提，不证明因果
    识别假设成立。
    """
    try:
        dataframe = _load_dataset(
            dataset_path
        )

        normalized_confounders = list(
            confounders or []
        )

        blocking_issues: list[str] = []
        warnings: list[str] = []

        thresholds = {
            "min_complete_cases": (
                min_complete_cases
            ),
            "min_group_size": min_group_size,
            "max_missing_rate": (
                max_missing_rate
            ),
        }

        if min_complete_cases < 2:
            raise ValueError(
                "min_complete_cases must be "
                "at least 2."
            )

        if min_group_size < 1:
            raise ValueError(
                "min_group_size must be "
                "at least 1."
            )

        if not 0 <= max_missing_rate < 1:
            raise ValueError(
                "max_missing_rate must be "
                "between 0 and 1."
            )

        # -------------------------------------------------
        # 第一层：变量配置检查
        # -------------------------------------------------

        if not treatment or not treatment.strip():
            blocking_issues.append(
                "Treatment must be specified."
            )

        if not outcome or not outcome.strip():
            blocking_issues.append(
                "Outcome must be specified."
            )

        if not normalized_confounders:
            blocking_issues.append(
                "At least one confounder is required."
            )

        duplicate_confounders = _find_duplicates(
            normalized_confounders
        )

        if duplicate_confounders:
            blocking_issues.append(
                "Confounders contain duplicate "
                f"variables: {duplicate_confounders}"
            )

        if (
            treatment
            and outcome
            and treatment == outcome
        ):
            blocking_issues.append(
                "Treatment and outcome must be "
                "different variables."
            )

        if treatment in normalized_confounders:
            blocking_issues.append(
                "Treatment cannot also be included "
                "as a confounder."
            )

        if outcome in normalized_confounders:
            blocking_issues.append(
                "Outcome cannot also be included "
                "as a confounder."
            )

        requested_variables = [
            treatment,
            outcome,
            *normalized_confounders,
        ]

        requested_variables = [
            variable
            for variable in requested_variables
            if variable
        ]

        unique_requested_variables = list(
            dict.fromkeys(requested_variables)
        )

        missing_columns = [
            variable
            for variable
            in unique_requested_variables
            if variable not in dataframe.columns
        ]

        if missing_columns:
            blocking_issues.append(
                "The following variables do not "
                f"exist: {missing_columns}"
            )

        base_sample_summary = {
            "total_rows": int(
                len(dataframe)
            ),
            "complete_cases": None,
            "dropped_rows": None,
            "treatment_counts": {},
            "treatment_rate": None,
        }

        if blocking_issues:
            return _build_result(
                ready=False,
                treatment=treatment,
                outcome=outcome,
                confounders=(
                    normalized_confounders
                ),
                blocking_issues=(
                    blocking_issues
                ),
                warnings=warnings,
                sample_summary=(
                    base_sample_summary
                ),
                variable_summary={},
                thresholds=thresholds,
            )

        # -------------------------------------------------
        # 第二层：变量类型检查
        # -------------------------------------------------

        if not is_numeric_dtype(
            dataframe[treatment]
        ):
            blocking_issues.append(
                f"Treatment '{treatment}' must be "
                "numeric for the current DML "
                "implementation."
            )

        if not is_numeric_dtype(
            dataframe[outcome]
        ):
            blocking_issues.append(
                f"Outcome '{outcome}' must be "
                "numeric for the current DML "
                "implementation."
            )

        for confounder in normalized_confounders:
            if not is_numeric_dtype(
                dataframe[confounder]
            ):
                blocking_issues.append(
                    f"Confounder '{confounder}' "
                    "must be numeric for the current "
                    "DML implementation."
                )

        if blocking_issues:
            return _build_result(
                ready=False,
                treatment=treatment,
                outcome=outcome,
                confounders=(
                    normalized_confounders
                ),
                blocking_issues=(
                    blocking_issues
                ),
                warnings=warnings,
                sample_summary=(
                    base_sample_summary
                ),
                variable_summary={},
                thresholds=thresholds,
            )

        # -------------------------------------------------
        # 第三层：缺失与非有限值处理
        # -------------------------------------------------

        selected_data = dataframe[
            unique_requested_variables
        ].copy()

        selected_data = selected_data.replace(
            [np.inf, -np.inf],
            np.nan,
        )

        missing_rates = {
            column: float(
                selected_data[column]
                .isna()
                .mean()
            )
            for column
            in unique_requested_variables
        }

        high_missing_variables = [
            column
            for column, missing_rate
            in missing_rates.items()
            if missing_rate > max_missing_rate
        ]

        if high_missing_variables:
            blocking_issues.append(
                "The following variables exceed "
                f"the maximum missing rate of "
                f"{max_missing_rate:.0%}: "
                f"{high_missing_variables}"
            )

        complete_data = (
            selected_data
            .dropna()
            .copy()
        )

        total_rows = len(selected_data)
        complete_cases = len(complete_data)
        dropped_rows = (
            total_rows - complete_cases
        )

        if complete_cases < min_complete_cases:
            blocking_issues.append(
                "Too few complete observations for "
                "the current DML workflow. "
                f"Required: {min_complete_cases}; "
                f"available: {complete_cases}."
            )

        if dropped_rows > 0:
            warnings.append(
                f"{dropped_rows} rows would be "
                "removed by complete-case analysis."
            )

        # 后续检查需要至少有数据。
        if complete_cases == 0:
            return _build_result(
                ready=False,
                treatment=treatment,
                outcome=outcome,
                confounders=(
                    normalized_confounders
                ),
                blocking_issues=(
                    blocking_issues
                ),
                warnings=warnings,
                sample_summary={
                    "total_rows": total_rows,
                    "complete_cases": 0,
                    "dropped_rows": dropped_rows,
                    "treatment_counts": {},
                    "treatment_rate": None,
                },
                variable_summary={
                    "missing_rates": missing_rates,
                },
                thresholds=thresholds,
            )

        # -------------------------------------------------
        # 第四层：处理变量检查
        # -------------------------------------------------

        treatment_values = (
            complete_data[treatment]
            .drop_duplicates()
            .tolist()
        )

        treatment_counts_series = (
            complete_data[treatment]
            .value_counts()
            .sort_index()
        )

        treatment_counts = {
            str(value): int(count)
            for value, count
            in treatment_counts_series.items()
        }

        if len(treatment_values) < 2:
            blocking_issues.append(
                f"Treatment '{treatment}' has no "
                "variation."
            )

        elif len(treatment_values) > 2:
            blocking_issues.append(
                f"Treatment '{treatment}' contains "
                f"{len(treatment_values)} values. "
                "The current Week 6 DML workflow "
                "supports binary treatment only."
            )

        treatment_rate: float | None = None

        if set(treatment_values) == {0, 1}:
            treatment_rate = float(
                complete_data[treatment].mean()
            )

            treated_count = int(
                (
                    complete_data[treatment] == 1
                ).sum()
            )
            control_count = int(
                (
                    complete_data[treatment] == 0
                ).sum()
            )

            if treated_count < min_group_size:
                blocking_issues.append(
                    "The treatment group is too "
                    "small. "
                    f"Required: {min_group_size}; "
                    f"available: {treated_count}."
                )

            if control_count < min_group_size:
                blocking_issues.append(
                    "The control group is too "
                    "small. "
                    f"Required: {min_group_size}; "
                    f"available: {control_count}."
                )

            if (
                treatment_rate < 0.10
                or treatment_rate > 0.90
            ):
                warnings.append(
                    "Treatment assignment is highly "
                    "imbalanced. Positivity or "
                    "overlap may be weak."
                )


        elif len(treatment_values) == 2:
            minimum_group_count = int(
                treatment_counts_series.min()
            )
            if minimum_group_count < min_group_size:
                blocking_issues.append(
                    "At least one treatment group "
                    "is too small. "
                    f"Required per group: "
                    f"{min_group_size}; smallest "
                    f"group: {minimum_group_count}."
                )
            blocking_issues.append(
                "Treatment must be coded as 0 and 1 "
                "for the current Week 6 DML workflow. "
                f"Observed values: {treatment_values}"
            )

        # -------------------------------------------------
        # 第五层：结果变量与混杂变量检查
        # -------------------------------------------------

        if complete_data[
            outcome
        ].nunique() < 2:
            blocking_issues.append(
                f"Outcome '{outcome}' has no "
                "variation."
            )

        constant_confounders = [
            confounder
            for confounder
            in normalized_confounders
            if complete_data[
                confounder
            ].nunique() < 2
        ]

        if constant_confounders:
            blocking_issues.append(
                "The following confounders have no "
                f"variation: {constant_confounders}"
            )

        variable_summary = {
            "missing_rates": missing_rates,
            "treatment_unique_values": [
                value.item()
                if isinstance(value, np.generic)
                else value
                for value in treatment_values
            ],
            "outcome_unique_count": int(
                complete_data[
                    outcome
                ].nunique()
            ),
            "constant_confounders": (
                constant_confounders
            ),
        }

        sample_summary = {
            "total_rows": int(total_rows),
            "complete_cases": int(
                complete_cases
            ),
            "dropped_rows": int(
                dropped_rows
            ),
            "treatment_counts": (
                treatment_counts
            ),
            "treatment_rate": (
                treatment_rate
            ),
        }

        # -------------------------------------------------
        # 第六层：无法由 CSV 验证的因果边界
        # -------------------------------------------------

        warnings.extend(
            [
                (
                    "Unobserved confounding cannot "
                    "be ruled out."
                ),
                (
                    "Causal identification depends "
                    "on assumptions that cannot be "
                    "verified from the CSV alone."
                ),
                (
                    "Readiness means that the data "
                    "meet the minimum technical "
                    "requirements; it does not prove "
                    "that a causal interpretation is "
                    "valid."
                ),
            ]
        )

        return _build_result(
            ready=not blocking_issues,
            treatment=treatment,
            outcome=outcome,
            confounders=normalized_confounders,
            blocking_issues=blocking_issues,
            warnings=warnings,
            sample_summary=sample_summary,
            variable_summary=variable_summary,
            thresholds=thresholds,
        )

    except ValueError as exc:
        return _error_response(
            str(exc)
        )
    except Exception as exc:
        return _error_response(
            "Causal readiness check failed: "
            f"{exc}"
        )