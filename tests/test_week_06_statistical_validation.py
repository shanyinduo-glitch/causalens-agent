from causalens.agent.nodes import (
    _validate_correlation_tool_call,
    _validate_group_comparison_tool_call,
    _validate_ols_tool_call,
)


DATASET_PATH = (
    "data/student_performance.csv"
)


def test_valid_correlation_call():
    errors = (
        _validate_correlation_tool_call(
            {
                "dataset_path": DATASET_PATH,
                "x_column": "study_hours",
                "y_column": "score",
                "method": "pearson",
            }
        )
    )

    assert errors == []


def test_correlation_rejects_same_column():
    errors = (
        _validate_correlation_tool_call(
            {
                "dataset_path": DATASET_PATH,
                "x_column": "score",
                "y_column": "score",
                "method": "pearson",
            }
        )
    )

    assert errors


def test_correlation_rejects_categorical_column():
    errors = (
        _validate_correlation_tool_call(
            {
                "dataset_path": DATASET_PATH,
                "x_column": "major",
                "y_column": "score",
                "method": "pearson",
            }
        )
    )

    assert any(
        "数值字段" in error
        for error in errors
    )


def test_valid_group_comparison_call():
    errors = (
        _validate_group_comparison_tool_call(
            {
                "dataset_path": DATASET_PATH,
                "group_column": "major",
                "value_column": "score",
            }
        )
    )

    assert errors == []


def test_group_comparison_rejects_text_value():
    errors = (
        _validate_group_comparison_tool_call(
            {
                "dataset_path": DATASET_PATH,
                "group_column": "passed",
                "value_column": "major",
            }
        )
    )

    assert any(
        "数值字段" in error
        for error in errors
    )


def test_valid_ols_call():
    errors = _validate_ols_tool_call(
        {
            "dataset_path": DATASET_PATH,
            "outcome": "score",
            "predictors": [
                "study_hours",
                "attendance_rate",
            ],
            "categorical_columns": [],
            "robust_covariance": "HC3",
        }
    )

    assert errors == []


def test_ols_accepts_categorical_predictor():
    errors = _validate_ols_tool_call(
        {
            "dataset_path": DATASET_PATH,
            "outcome": "score",
            "predictors": [
                "study_hours",
                "major",
            ],
            "categorical_columns": [
                "major",
            ],
            "robust_covariance": "HC3",
        }
    )

    assert errors == []


def test_ols_rejects_outcome_in_predictors():
    errors = _validate_ols_tool_call(
        {
            "dataset_path": DATASET_PATH,
            "outcome": "score",
            "predictors": [
                "score",
                "study_hours",
            ],
            "categorical_columns": [],
            "robust_covariance": "HC3",
        }
    )

    assert errors


def test_ols_rejects_unknown_categorical():
    errors = _validate_ols_tool_call(
        {
            "dataset_path": DATASET_PATH,
            "outcome": "score",
            "predictors": [
                "study_hours",
            ],
            "categorical_columns": [
                "major",
            ],
            "robust_covariance": "HC3",
        }
    )

    assert any(
        "predictors 的子集" in error
        for error in errors
    )


def test_ols_rejects_invalid_covariance():
    errors = _validate_ols_tool_call(
        {
            "dataset_path": DATASET_PATH,
            "outcome": "score",
            "predictors": [
                "study_hours",
            ],
            "categorical_columns": [],
            "robust_covariance": "invalid",
        }
    )

    assert any(
        "HC0" in error
        for error in errors
    )