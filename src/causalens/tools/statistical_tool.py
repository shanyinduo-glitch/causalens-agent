from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from scipy.stats import f_oneway, pearsonr, spearmanr, ttest_ind

from causalens.schemas import StatisticalResult


SUPPORTED_CORRELATION_METHODS = {
    "pearson",
    "spearman",
}


def _success_response(
    result: StatisticalResult,
) -> dict[str, Any]:
    """生成与现有工具风格一致的成功结果。"""
    return {
        "success": True,
        "result": result.model_dump(),
    }


def _error_response(
    message: str,
) -> dict[str, Any]:
    """生成与现有工具风格一致的失败结果。"""
    return {
        "success": False,
        "error": message,
    }


def _load_dataset(
    dataset_path: str,
) -> pd.DataFrame:
    """读取 CSV，并将底层异常转换成更易理解的错误。"""
    if not dataset_path or not dataset_path.strip():
        raise ValueError("dataset_path cannot be empty.")

    try:
        dataframe = pd.read_csv(dataset_path)
    except FileNotFoundError as exc:
        raise ValueError(
            f"Dataset file does not exist: {dataset_path}"
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


def _validate_columns_exist(
    dataframe: pd.DataFrame,
    columns: list[str],
) -> None:
    """验证用户指定的字段是否全部存在。"""
    missing_columns = [
        column
        for column in columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            "The following columns do not exist: "
            f"{missing_columns}"
        )


def _validate_numeric_column(
    dataframe: pd.DataFrame,
    column: str,
) -> None:
    """验证字段是否为数值型。"""
    if not is_numeric_dtype(dataframe[column]):
        raise ValueError(
            f"Column '{column}' must be numeric."
        )


def _to_python_value(
    value: Any,
) -> Any:
    """将 NumPy 标量转换为普通 Python 标量。"""
    if isinstance(value, np.generic):
        return value.item()

    return value


def run_correlation(
    dataset_path: str,
    x_column: str,
    y_column: str,
    method: str = "pearson",
) -> dict[str, Any]:
    """
    对两个数值字段执行 Pearson 或 Spearman 相关分析。

    Pearson:
        衡量线性相关关系。

    Spearman:
        基于秩，衡量单调关系。
    """
    try:
        normalized_method = method.strip().lower()

        if normalized_method not in SUPPORTED_CORRELATION_METHODS:
            raise ValueError(
                "Unsupported correlation method. "
                "Expected one of: "
                f"{sorted(SUPPORTED_CORRELATION_METHODS)}"
            )

        if x_column == y_column:
            raise ValueError(
                "x_column and y_column must be different."
            )

        dataframe = _load_dataset(dataset_path)

        _validate_columns_exist(
            dataframe=dataframe,
            columns=[x_column, y_column],
        )

        _validate_numeric_column(
            dataframe=dataframe,
            column=x_column,
        )
        _validate_numeric_column(
            dataframe=dataframe,
            column=y_column,
        )

        selected_data = dataframe[
            [x_column, y_column]
        ].copy()

        total_rows = len(selected_data)

        complete_data = selected_data.dropna()

        sample_size = len(complete_data)
        removed_rows = total_rows - sample_size

        if sample_size < 3:
            raise ValueError(
                "Correlation analysis requires at least "
                "3 complete observations."
            )

        if complete_data[x_column].nunique() < 2:
            raise ValueError(
                f"Column '{x_column}' has no variation."
            )

        if complete_data[y_column].nunique() < 2:
            raise ValueError(
                f"Column '{y_column}' has no variation."
            )

        x_values = complete_data[x_column].to_numpy()
        y_values = complete_data[y_column].to_numpy()

        if normalized_method == "pearson":
            correlation_result = pearsonr(
                x_values,
                y_values,
            )
        else:
            correlation_result = spearmanr(
                x_values,
                y_values,
            )

        estimate = float(
            correlation_result.statistic
        )
        p_value = float(
            correlation_result.pvalue
        )

        warnings = [
            "Correlation does not establish causality."
        ]

        if removed_rows > 0:
            warnings.append(
                f"{removed_rows} rows were removed "
                "because of missing values."
            )

        if sample_size < 30:
            warnings.append(
                "The analysis uses fewer than 30 "
                "complete observations."
            )

        result = StatisticalResult(
            analysis_type="correlation",
            variables=[x_column, y_column],
            sample_size=sample_size,
            estimate=estimate,
            p_value=p_value,
            details={
                "method": normalized_method,
                "complete_case_analysis": True,
                "removed_rows": removed_rows,
                "x_summary": {
                    "mean": float(
                        complete_data[x_column].mean()
                    ),
                    "std": float(
                        complete_data[x_column].std()
                    ),
                    "min": float(
                        complete_data[x_column].min()
                    ),
                    "max": float(
                        complete_data[x_column].max()
                    ),
                },
                "y_summary": {
                    "mean": float(
                        complete_data[y_column].mean()
                    ),
                    "std": float(
                        complete_data[y_column].std()
                    ),
                    "min": float(
                        complete_data[y_column].min()
                    ),
                    "max": float(
                        complete_data[y_column].max()
                    ),
                },
            },
            warnings=warnings,
        )

        return _success_response(result)

    except ValueError as exc:
        return _error_response(str(exc))
    except Exception as exc:
        return _error_response(
            f"Correlation analysis failed: {exc}"
        )


def run_group_comparison(
    dataset_path: str,
    group_column: str,
    value_column: str,
) -> dict[str, Any]:
    """
    比较不同组的数值型结果。

    两组：
        Welch independent-samples t-test。

    三组及以上：
        One-way ANOVA。
    """
    try:
        if group_column == value_column:
            raise ValueError(
                "group_column and value_column "
                "must be different."
            )

        dataframe = _load_dataset(dataset_path)

        _validate_columns_exist(
            dataframe=dataframe,
            columns=[group_column, value_column],
        )

        _validate_numeric_column(
            dataframe=dataframe,
            column=value_column,
        )

        selected_data = dataframe[
            [group_column, value_column]
        ].copy()

        total_rows = len(selected_data)

        complete_data = selected_data.dropna()

        sample_size = len(complete_data)
        removed_rows = total_rows - sample_size

        if sample_size < 4:
            raise ValueError(
                "Group comparison requires at least "
                "4 complete observations."
            )

        group_values = (
            complete_data[group_column]
            .drop_duplicates()
            .tolist()
        )

        if len(group_values) < 2:
            raise ValueError(
                "Group comparison requires at least "
                "2 distinct groups."
            )

        grouped_arrays: list[np.ndarray] = []
        group_summary: list[dict[str, Any]] = []

        for group_value in group_values:
            group_data = complete_data.loc[
                complete_data[group_column] == group_value,
                value_column,
            ].to_numpy(dtype=float)

            if len(group_data) < 2:
                raise ValueError(
                    "Each group must contain at least "
                    "2 complete observations. "
                    f"Group {group_value!r} contains "
                    f"{len(group_data)}."
                )

            grouped_arrays.append(group_data)

            group_summary.append(
                {
                    "group": _to_python_value(
                        group_value
                    ),
                    "sample_size": int(
                        len(group_data)
                    ),
                    "mean": float(
                        np.mean(group_data)
                    ),
                    "std": float(
                        np.std(
                            group_data,
                            ddof=1,
                        )
                    ),
                    "min": float(
                        np.min(group_data)
                    ),
                    "max": float(
                        np.max(group_data)
                    ),
                }
            )

        warnings = [
            "Group differences do not establish causality."
        ]

        if removed_rows > 0:
            warnings.append(
                f"{removed_rows} rows were removed "
                "because of missing values."
            )

        if len(group_values) == 2:
            test_result = ttest_ind(
                grouped_arrays[0],
                grouped_arrays[1],
                equal_var=False,
                nan_policy="omit",
            )

            first_mean = float(
                np.mean(grouped_arrays[0])
            )
            second_mean = float(
                np.mean(grouped_arrays[1])
            )

            result = StatisticalResult(
                analysis_type="welch_t_test",
                variables=[
                    group_column,
                    value_column,
                ],
                sample_size=sample_size,
                estimate=first_mean - second_mean,
                statistic=float(
                    test_result.statistic
                ),
                p_value=float(
                    test_result.pvalue
                ),
                details={
                    "test": "Welch independent-samples t-test",
                    "number_of_groups": 2,
                    "mean_difference_definition": (
                        "first_group_mean - "
                        "second_group_mean"
                    ),
                    "group_summary": group_summary,
                    "complete_case_analysis": True,
                    "removed_rows": removed_rows,
                },
                warnings=[
                    *warnings,
                    (
                        "Welch's t-test does not assume "
                        "equal group variances."
                    ),
                ],
            )

            return _success_response(result)

        test_result = f_oneway(
            *grouped_arrays
        )

        result = StatisticalResult(
            analysis_type="one_way_anova",
            variables=[
                group_column,
                value_column,
            ],
            sample_size=sample_size,
            statistic=float(
                test_result.statistic
            ),
            p_value=float(
                test_result.pvalue
            ),
            details={
                "test": "One-way ANOVA",
                "number_of_groups": len(
                    group_values
                ),
                "group_summary": group_summary,
                "complete_case_analysis": True,
                "removed_rows": removed_rows,
            },
            warnings=[
                *warnings,
                (
                    "One-way ANOVA assumes independent "
                    "observations, approximately normal "
                    "within-group residuals, and similar "
                    "within-group variances."
                ),
            ],
        )

        return _success_response(result)

    except ValueError as exc:
        return _error_response(str(exc))
    except Exception as exc:
        return _error_response(
            f"Group comparison failed: {exc}"
        )