import pandas as pd
import pytest

from causalens.tools.statistical_tool import (
    run_correlation,
    run_group_comparison,
)


def save_test_data(
    tmp_path,
    dataframe: pd.DataFrame,
) -> str:
    dataset_path = tmp_path / "test_data.csv"
    dataframe.to_csv(
        dataset_path,
        index=False,
    )
    return str(dataset_path)


def test_run_pearson_correlation_success(
    tmp_path,
):
    dataset_path = save_test_data(
        tmp_path,
        pd.DataFrame(
            {
                "x": [1, 2, 3, 4, 5],
                "y": [2, 4, 6, 8, 10],
            }
        ),
    )

    response = run_correlation(
        dataset_path=dataset_path,
        x_column="x",
        y_column="y",
        method="pearson",
    )

    assert response["success"] is True

    result = response["result"]

    assert result["analysis_type"] == "correlation"
    assert result["sample_size"] == 5
    assert result["estimate"] == pytest.approx(1.0)
    assert result["details"]["method"] == "pearson"


def test_run_spearman_correlation_success(
    tmp_path,
):
    dataset_path = save_test_data(
        tmp_path,
        pd.DataFrame(
            {
                "x": [1, 2, 3, 4, 5],
                "y": [10, 8, 6, 4, 2],
            }
        ),
    )

    response = run_correlation(
        dataset_path=dataset_path,
        x_column="x",
        y_column="y",
        method="spearman",
    )

    assert response["success"] is True
    assert response["result"]["estimate"] == pytest.approx(
        -1.0
    )


def test_correlation_rejects_unknown_method(
    tmp_path,
):
    dataset_path = save_test_data(
        tmp_path,
        pd.DataFrame(
            {
                "x": [1, 2, 3],
                "y": [2, 3, 4],
            }
        ),
    )

    response = run_correlation(
        dataset_path=dataset_path,
        x_column="x",
        y_column="y",
        method="kendall",
    )

    assert response["success"] is False
    assert "Unsupported correlation method" in response["error"]


def test_correlation_rejects_missing_column(
    tmp_path,
):
    dataset_path = save_test_data(
        tmp_path,
        pd.DataFrame(
            {
                "x": [1, 2, 3],
                "y": [2, 3, 4],
            }
        ),
    )

    response = run_correlation(
        dataset_path=dataset_path,
        x_column="x",
        y_column="unknown",
    )

    assert response["success"] is False
    assert "do not exist" in response["error"]


def test_correlation_rejects_non_numeric_column(
    tmp_path,
):
    dataset_path = save_test_data(
        tmp_path,
        pd.DataFrame(
            {
                "x": ["a", "b", "c"],
                "y": [1, 2, 3],
            }
        ),
    )

    response = run_correlation(
        dataset_path=dataset_path,
        x_column="x",
        y_column="y",
    )

    assert response["success"] is False
    assert "must be numeric" in response["error"]


def test_correlation_rejects_constant_column(
    tmp_path,
):
    dataset_path = save_test_data(
        tmp_path,
        pd.DataFrame(
            {
                "x": [1, 1, 1, 1],
                "y": [1, 2, 3, 4],
            }
        ),
    )

    response = run_correlation(
        dataset_path=dataset_path,
        x_column="x",
        y_column="y",
    )

    assert response["success"] is False
    assert "has no variation" in response["error"]


def test_group_comparison_runs_welch_t_test(
    tmp_path,
):
    dataset_path = save_test_data(
        tmp_path,
        pd.DataFrame(
            {
                "group": [
                    "control",
                    "control",
                    "control",
                    "treated",
                    "treated",
                    "treated",
                ],
                "value": [
                    1.0,
                    2.0,
                    3.0,
                    5.0,
                    6.0,
                    7.0,
                ],
            }
        ),
    )

    response = run_group_comparison(
        dataset_path=dataset_path,
        group_column="group",
        value_column="value",
    )

    assert response["success"] is True

    result = response["result"]

    assert result["analysis_type"] == "welch_t_test"
    assert result["sample_size"] == 6
    assert result["details"]["number_of_groups"] == 2
    assert len(result["details"]["group_summary"]) == 2
    assert result["p_value"] < 0.05


def test_group_comparison_runs_one_way_anova(
    tmp_path,
):
    dataset_path = save_test_data(
        tmp_path,
        pd.DataFrame(
            {
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
                "value": [
                    1.0,
                    2.0,
                    3.0,
                    5.0,
                    6.0,
                    7.0,
                    10.0,
                    11.0,
                    12.0,
                ],
            }
        ),
    )

    response = run_group_comparison(
        dataset_path=dataset_path,
        group_column="group",
        value_column="value",
    )

    assert response["success"] is True

    result = response["result"]

    assert result["analysis_type"] == "one_way_anova"
    assert result["sample_size"] == 9
    assert result["details"]["number_of_groups"] == 3
    assert result["p_value"] < 0.05


def test_group_comparison_rejects_single_group(
    tmp_path,
):
    dataset_path = save_test_data(
        tmp_path,
        pd.DataFrame(
            {
                "group": ["a", "a", "a", "a"],
                "value": [1.0, 2.0, 3.0, 4.0],
            }
        ),
    )

    response = run_group_comparison(
        dataset_path=dataset_path,
        group_column="group",
        value_column="value",
    )

    assert response["success"] is False
    assert "at least 2 distinct groups" in response["error"]


def test_group_comparison_rejects_small_group(
    tmp_path,
):
    dataset_path = save_test_data(
        tmp_path,
        pd.DataFrame(
            {
                "group": ["a", "a", "a", "b"],
                "value": [1.0, 2.0, 3.0, 4.0],
            }
        ),
    )

    response = run_group_comparison(
        dataset_path=dataset_path,
        group_column="group",
        value_column="value",
    )

    assert response["success"] is False
    assert "Each group must contain at least 2" in response["error"]