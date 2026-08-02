import pandas as pd
import pytest

from causalens.tools.regression_tool import (
    run_ols,
)


def save_test_data(
    tmp_path,
    dataframe: pd.DataFrame,
) -> str:
    dataset_path = (
        tmp_path / "regression_data.csv"
    )

    dataframe.to_csv(
        dataset_path,
        index=False,
    )

    return str(dataset_path)


def test_run_ols_with_numeric_predictors(
    tmp_path,
):
    dataframe = pd.DataFrame(
        {
            "x1": [
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
            ],
            "x2": [
                2,
                1,
                3,
                2,
                5,
                4,
                7,
                6,
            ],
            "y": [
                5,
                6,
                10,
                11,
                17,
                18,
                24,
                25,
            ],
        }
    )

    dataset_path = save_test_data(
        tmp_path,
        dataframe,
    )

    response = run_ols(
        dataset_path=dataset_path,
        outcome="y",
        predictors=["x1", "x2"],
    )

    assert response["success"] is True

    result = response["result"]

    assert result["analysis_type"] == "ols"
    assert result["sample_size"] == 8
    assert result["covariance_type"] == "HC3"
    assert len(result["coefficients"]) == 3
    assert result["r_squared"] > 0.90


def test_run_ols_with_categorical_predictor(
    tmp_path,
):
    dataframe = pd.DataFrame(
        {
            "x": [
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
            ],
            "group": [
                "a",
                "a",
                "a",
                "b",
                "b",
                "b",
                "c",
                "c",
                "c",
            ],
            "y": [
                2,
                4,
                6,
                9,
                11,
                13,
                16,
                18,
                20,
            ],
        }
    )

    dataset_path = save_test_data(
        tmp_path,
        dataframe,
    )

    response = run_ols(
        dataset_path=dataset_path,
        outcome="y",
        predictors=["x", "group"],
        categorical_columns=["group"],
    )

    assert response["success"] is True

    result = response["result"]

    coefficient_terms = {
        coefficient["term"]
        for coefficient in result[
            "coefficients"
        ]
    }

    assert "const" in coefficient_terms
    assert "x" in coefficient_terms
    assert "group_b" in coefficient_terms
    assert "group_c" in coefficient_terms

    reference_categories = result[
        "diagnostics"
    ]["design_matrix"][
        "reference_categories"
    ]

    assert reference_categories[
        "group"
    ] == "a"


def test_run_ols_rejects_missing_column(
    tmp_path,
):
    dataset_path = save_test_data(
        tmp_path,
        pd.DataFrame(
            {
                "x": [1, 2, 3, 4],
                "y": [2, 4, 6, 8],
            }
        ),
    )

    response = run_ols(
        dataset_path=dataset_path,
        outcome="y",
        predictors=["unknown"],
    )

    assert response["success"] is False
    assert "do not exist" in response["error"]


def test_run_ols_rejects_non_numeric_outcome(
    tmp_path,
):
    dataset_path = save_test_data(
        tmp_path,
        pd.DataFrame(
            {
                "x": [1, 2, 3, 4],
                "y": ["a", "b", "c", "d"],
            }
        ),
    )

    response = run_ols(
        dataset_path=dataset_path,
        outcome="y",
        predictors=["x"],
    )

    assert response["success"] is False
    assert "must be numeric" in response["error"]


def test_run_ols_rejects_unmarked_text_predictor(
    tmp_path,
):
    dataset_path = save_test_data(
        tmp_path,
        pd.DataFrame(
            {
                "group": [
                    "a",
                    "a",
                    "b",
                    "b",
                ],
                "y": [1, 2, 4, 5],
            }
        ),
    )

    response = run_ols(
        dataset_path=dataset_path,
        outcome="y",
        predictors=["group"],
    )

    assert response["success"] is False
    assert (
        "numeric or included in "
        "categorical_columns"
    ) in response["error"]


def test_run_ols_rejects_constant_outcome(
    tmp_path,
):
    dataset_path = save_test_data(
        tmp_path,
        pd.DataFrame(
            {
                "x": [1, 2, 3, 4, 5],
                "y": [10, 10, 10, 10, 10],
            }
        ),
    )

    response = run_ols(
        dataset_path=dataset_path,
        outcome="y",
        predictors=["x"],
    )

    assert response["success"] is False
    assert "has no variation" in response["error"]


def test_run_ols_rejects_constant_predictor(
    tmp_path,
):
    dataset_path = save_test_data(
        tmp_path,
        pd.DataFrame(
            {
                "x": [1, 1, 1, 1, 1],
                "y": [1, 2, 3, 4, 5],
            }
        ),
    )

    response = run_ols(
        dataset_path=dataset_path,
        outcome="y",
        predictors=["x"],
    )

    assert response["success"] is False
    assert "has no variation" in response["error"]


def test_run_ols_rejects_perfect_collinearity(
    tmp_path,
):
    dataset_path = save_test_data(
        tmp_path,
        pd.DataFrame(
            {
                "x1": [1, 2, 3, 4, 5, 6],
                "x2": [2, 4, 6, 8, 10, 12],
                "y": [3, 5, 7, 9, 11, 13],
            }
        ),
    )

    response = run_ols(
        dataset_path=dataset_path,
        outcome="y",
        predictors=["x1", "x2"],
    )

    assert response["success"] is False
    assert "rank deficient" in response["error"]


def test_run_ols_records_dropped_rows(
    tmp_path,
):
    dataset_path = save_test_data(
        tmp_path,
        pd.DataFrame(
            {
                "x": [
                    1.0,
                    2.0,
                    None,
                    4.0,
                    5.0,
                    6.0,
                ],
                "y": [
                    2.0,
                    4.0,
                    6.0,
                    8.0,
                    10.0,
                    12.0,
                ],
            }
        ),
    )

    response = run_ols(
        dataset_path=dataset_path,
        outcome="y",
        predictors=["x"],
    )

    assert response["success"] is True
    assert response["result"][
        "sample_size"
    ] == 5
    assert response["result"][
        "dropped_rows"
    ] == 1


def test_run_ols_rejects_unknown_covariance(
    tmp_path,
):
    dataset_path = save_test_data(
        tmp_path,
        pd.DataFrame(
            {
                "x": [1, 2, 3, 4, 5],
                "y": [2, 3, 5, 7, 11],
            }
        ),
    )

    response = run_ols(
        dataset_path=dataset_path,
        outcome="y",
        predictors=["x"],
        robust_covariance="UNKNOWN",
    )

    assert response["success"] is False
    assert (
        "Unsupported robust covariance type"
        in response["error"]
    )