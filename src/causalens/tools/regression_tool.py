from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from pandas.api.types import is_numeric_dtype
from statsmodels.stats.diagnostic import het_breuschpagan

from causalens.schemas import (
    RegressionCoefficient,
    RegressionResult,
)


SUPPORTED_ROBUST_COVARIANCE_TYPES = {
    "HC0",
    "HC1",
    "HC2",
    "HC3",
}


def _success_response(
    result: RegressionResult,
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


def _validate_unique_values(
    values: list[str],
    argument_name: str,
) -> None:
    duplicates = sorted(
        {
            value
            for value in values
            if values.count(value) > 1
        }
    )

    if duplicates:
        raise ValueError(
            f"{argument_name} contains duplicate "
            f"columns: {duplicates}"
        )


def _validate_model_arguments(
    dataframe: pd.DataFrame,
    outcome: str,
    predictors: list[str],
    categorical_columns: list[str],
    robust_covariance: str,
) -> None:
    if not outcome or not outcome.strip():
        raise ValueError(
            "outcome cannot be empty."
        )

    if not predictors:
        raise ValueError(
            "At least one predictor is required."
        )

    _validate_unique_values(
        values=predictors,
        argument_name="predictors",
    )

    _validate_unique_values(
        values=categorical_columns,
        argument_name="categorical_columns",
    )

    if outcome in predictors:
        raise ValueError(
            "outcome cannot also be used as a "
            "predictor."
        )

    invalid_categorical_columns = [
        column
        for column in categorical_columns
        if column not in predictors
    ]

    if invalid_categorical_columns:
        raise ValueError(
            "categorical_columns must be a subset "
            "of predictors. Invalid columns: "
            f"{invalid_categorical_columns}"
        )

    required_columns = [
        outcome,
        *predictors,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            "The following columns do not exist: "
            f"{missing_columns}"
        )

    if robust_covariance not in (
        SUPPORTED_ROBUST_COVARIANCE_TYPES
    ):
        raise ValueError(
            "Unsupported robust covariance type. "
            "Expected one of: "
            f"{sorted(SUPPORTED_ROBUST_COVARIANCE_TYPES)}"
        )

    if not is_numeric_dtype(
        dataframe[outcome]
    ):
        raise ValueError(
            f"Outcome column '{outcome}' "
            "must be numeric."
        )

    for predictor in predictors:
        if predictor in categorical_columns:
            continue

        if not is_numeric_dtype(
            dataframe[predictor]
        ):
            raise ValueError(
                f"Predictor '{predictor}' must be "
                "numeric or included in "
                "categorical_columns."
            )


def _prepare_complete_cases(
    dataframe: pd.DataFrame,
    required_columns: list[str],
) -> tuple[
    pd.DataFrame,
    dict[str, float],
    int,
]:
    selected_data = dataframe[
        required_columns
    ].copy()

    missing_rates = {
        column: float(
            selected_data[column]
            .isna()
            .mean()
        )
        for column in required_columns
    }

    complete_data = selected_data.dropna()

    dropped_rows = (
        len(selected_data)
        - len(complete_data)
    )

    if complete_data.empty:
        raise ValueError(
            "No complete observations remain after "
            "removing rows with missing values."
        )

    return (
        complete_data,
        missing_rates,
        dropped_rows,
    )


def _build_design_matrix(
    complete_data: pd.DataFrame,
    predictors: list[str],
    categorical_columns: list[str],
) -> tuple[
    pd.DataFrame,
    dict[str, Any],
]:
    design_parts: list[pd.DataFrame] = []

    reference_categories: dict[str, Any] = {}
    category_levels: dict[str, list[Any]] = {}

    for predictor in predictors:
        series = complete_data[predictor]

        if predictor not in categorical_columns:
            if series.nunique() < 2:
                raise ValueError(
                    f"Predictor '{predictor}' "
                    "has no variation."
                )

            numeric_part = pd.DataFrame(
                {
                    predictor: series.astype(float)
                },
                index=complete_data.index,
            )

            design_parts.append(
                numeric_part
            )

            continue

        categories = sorted(
            series.drop_duplicates().tolist(),
            key=lambda value: str(value),
        )

        if len(categories) < 2:
            raise ValueError(
                f"Categorical predictor "
                f"'{predictor}' must contain at "
                "least 2 levels."
            )

        reference_categories[predictor] = (
            categories[0]
        )
        category_levels[predictor] = categories

        categorical_series = pd.Series(
            pd.Categorical(
                series,
                categories=categories,
            ),
            index=complete_data.index,
            name=predictor,
        )

        dummy_variables = pd.get_dummies(
            categorical_series,
            prefix=predictor,
            drop_first=True,
            dtype=float,
        )

        design_parts.append(
            dummy_variables
        )

    if not design_parts:
        raise ValueError(
            "The model contains no usable "
            "predictors."
        )

    design_matrix = pd.concat(
        design_parts,
        axis=1,
    )

    design_matrix = sm.add_constant(
        design_matrix,
        has_constant="add",
    )

    metadata = {
        "reference_categories": (
            reference_categories
        ),
        "category_levels": category_levels,
        "design_columns": (
            design_matrix.columns.tolist()
        ),
    }

    return design_matrix, metadata


def _check_design_matrix(
    design_matrix: pd.DataFrame,
    sample_size: int,
) -> None:
    number_of_parameters = (
        design_matrix.shape[1]
    )

    if sample_size <= number_of_parameters:
        raise ValueError(
            "The number of complete observations "
            "must be greater than the number of "
            "estimated parameters. "
            f"Observations: {sample_size}; "
            f"parameters: {number_of_parameters}."
        )

    matrix_rank = np.linalg.matrix_rank(
        design_matrix.to_numpy(
            dtype=float
        )
    )

    if matrix_rank < number_of_parameters:
        raise ValueError(
            "The regression design matrix is rank "
            "deficient. This may be caused by "
            "perfect multicollinearity or duplicate "
            "predictors."
        )


def _build_coefficients(
    term_names: list[str],
    ordinary_results: Any,
    robust_results: Any,
) -> list[RegressionCoefficient]:
    robust_confidence_intervals = (
        np.asarray(
            robust_results.conf_int(
                alpha=0.05
            )
        )
    )

    coefficients: list[
        RegressionCoefficient
    ] = []

    for index, term_name in enumerate(
        term_names
    ):
        coefficients.append(
            RegressionCoefficient(
                term=term_name,
                estimate=float(
                    ordinary_results.params.iloc[
                        index
                    ]
                ),
                standard_error=float(
                    ordinary_results.bse.iloc[
                        index
                    ]
                ),
                robust_standard_error=float(
                    np.asarray(
                        robust_results.bse
                    )[index]
                ),
                p_value=float(
                    np.asarray(
                        robust_results.pvalues
                    )[index]
                ),
                confidence_interval=[
                    float(
                        robust_confidence_intervals[
                            index,
                            0,
                        ]
                    ),
                    float(
                        robust_confidence_intervals[
                            index,
                            1,
                        ]
                    ),
                ],
            )
        )

    return coefficients


def run_ols(
    dataset_path: str,
    outcome: str,
    predictors: list[str],
    categorical_columns: list[str] | None = None,
    robust_covariance: str = "HC3",
) -> dict[str, Any]:
    """
    执行带有异方差稳健标准误的 OLS 回归。

    回归系数与普通 OLS 相同，但 p 值、标准误和
    置信区间使用指定的稳健协方差估计。
    """
    try:
        normalized_covariance = (
            robust_covariance
            .strip()
            .upper()
        )

        normalized_categorical_columns = (
            list(categorical_columns or [])
        )

        dataframe = _load_dataset(
            dataset_path
        )

        _validate_model_arguments(
            dataframe=dataframe,
            outcome=outcome,
            predictors=predictors,
            categorical_columns=(
                normalized_categorical_columns
            ),
            robust_covariance=(
                normalized_covariance
            ),
        )

        required_columns = [
            outcome,
            *predictors,
        ]

        (
            complete_data,
            missing_rates,
            dropped_rows,
        ) = _prepare_complete_cases(
            dataframe=dataframe,
            required_columns=required_columns,
        )

        sample_size = len(
            complete_data
        )

        if complete_data[
            outcome
        ].nunique() < 2:
            raise ValueError(
                f"Outcome column '{outcome}' "
                "has no variation."
            )

        outcome_values = (
            complete_data[outcome]
            .astype(float)
        )

        (
            design_matrix,
            design_metadata,
        ) = _build_design_matrix(
            complete_data=complete_data,
            predictors=predictors,
            categorical_columns=(
                normalized_categorical_columns
            ),
        )

        _check_design_matrix(
            design_matrix=design_matrix,
            sample_size=sample_size,
        )

        model = sm.OLS(
            endog=outcome_values,
            exog=design_matrix,
        )

        ordinary_results = model.fit()

        robust_results = (
            ordinary_results
            .get_robustcov_results(
                cov_type=normalized_covariance
            )
        )

        coefficients = _build_coefficients(
            term_names=(
                design_matrix.columns.tolist()
            ),
            ordinary_results=ordinary_results,
            robust_results=robust_results,
        )

        bp_lm_statistic: float | None = None
        bp_lm_p_value: float | None = None
        bp_f_statistic: float | None = None
        bp_f_p_value: float | None = None
        bp_error: str | None = None

        try:
            (
                bp_lm_statistic,
                bp_lm_p_value,
                bp_f_statistic,
                bp_f_p_value,
            ) = het_breuschpagan(
                ordinary_results.resid,
                ordinary_results.model.exog,
            )

            bp_lm_statistic = float(
                bp_lm_statistic
            )
            bp_lm_p_value = float(
                bp_lm_p_value
            )
            bp_f_statistic = float(
                bp_f_statistic
            )
            bp_f_p_value = float(
                bp_f_p_value
            )

        except Exception as exc:
            bp_error = str(exc)

        condition_number = float(
            ordinary_results.condition_number
        )

        number_of_parameters = int(
            design_matrix.shape[1]
        )

        warnings = [
            (
                "Regression estimates are "
                "associations unless a valid causal "
                "design is established."
            ),
            (
                f"{normalized_covariance} robust "
                "standard errors were used for "
                "inference."
            ),
        ]

        if dropped_rows > 0:
            warnings.append(
                f"{dropped_rows} rows were removed "
                "because of missing values."
            )

        high_missing_columns = [
            column
            for column, missing_rate
            in missing_rates.items()
            if missing_rate > 0.20
        ]

        if high_missing_columns:
            warnings.append(
                "The following variables have more "
                "than 20% missing values: "
                f"{high_missing_columns}"
            )

        if sample_size < 30:
            warnings.append(
                "The regression uses fewer than 30 "
                "complete observations."
            )

        if sample_size < (
            10 * number_of_parameters
        ):
            warnings.append(
                "The number of observations is "
                "small relative to the number of "
                "estimated parameters."
            )

        if (
            bp_lm_p_value is not None
            and bp_lm_p_value < 0.05
        ):
            warnings.append(
                "The Breusch-Pagan test suggests "
                "possible heteroskedasticity."
            )

        if bp_error is not None:
            warnings.append(
                "The Breusch-Pagan diagnostic could "
                "not be computed, but the regression "
                "result was retained."
            )

        if condition_number > 1_000_000:
            warnings.append(
                "The design matrix has a large "
                "condition number. This may indicate "
                "scaling problems or "
                "multicollinearity."
            )

        result = RegressionResult(
            outcome=outcome,
            predictors=predictors,
            categorical_columns=(
                normalized_categorical_columns
            ),
            sample_size=sample_size,
            dropped_rows=dropped_rows,
            covariance_type=(
                normalized_covariance
            ),
            coefficients=coefficients,
            r_squared=float(
                ordinary_results.rsquared
            ),
            adjusted_r_squared=float(
                ordinary_results.rsquared_adj
            ),
            diagnostics={
                "total_rows": int(
                    len(dataframe)
                ),
                "number_of_parameters": (
                    number_of_parameters
                ),
                "missing_rates": (
                    missing_rates
                ),
                "condition_number": (
                    condition_number
                ),
                "design_matrix": (
                    design_metadata
                ),
                "breusch_pagan": {
                    "lm_statistic": (
                        bp_lm_statistic
                    ),
                    "lm_p_value": (
                        bp_lm_p_value
                    ),
                    "f_statistic": (
                        bp_f_statistic
                    ),
                    "f_p_value": (
                        bp_f_p_value
                    ),
                    "error": bp_error,
                },
            },
            warnings=warnings,
        )

        return _success_response(
            result
        )

    except ValueError as exc:
        return _error_response(
            str(exc)
        )
    except np.linalg.LinAlgError as exc:
        return _error_response(
            "OLS regression failed because of a "
            f"linear algebra error: {exc}"
        )
    except Exception as exc:
        return _error_response(
            f"OLS regression failed: {exc}"
        )