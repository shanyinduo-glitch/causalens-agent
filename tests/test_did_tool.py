import numpy as np
import pandas as pd

from causalens.tools.did_tool import (
    check_did_readiness,
    run_difference_in_differences,
)
from causalens.tools.causal_tool import run_causal_estimators


def _save(tmp_path, dataframe: pd.DataFrame) -> str:
    path = tmp_path / "did.csv"
    dataframe.to_csv(path, index=False)
    return str(path)


def _panel_data(unit_count: int = 240) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = []
    for unit_id in range(unit_count):
        treated = int(unit_id >= unit_count // 2)
        baseline = rng.normal(0, 1)
        unit_effect = rng.normal(0, 1)
        for post in (0, 1):
            outcome = (
                5
                + unit_effect
                + 0.8 * post
                + 2.5 * treated * post
                + 0.6 * baseline
                + rng.normal(0, 0.8)
            )
            rows.append(
                {
                    "unit_id": unit_id,
                    "treated_group": treated,
                    "post": post,
                    "outcome": outcome,
                    "baseline": baseline,
                }
            )
    return pd.DataFrame(rows)


def _repeated_cross_section_data(n_per_cell: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    rows = []
    for treated in (0, 1):
        for post in (0, 1):
            covariate = rng.normal(0, 1, n_per_cell)
            outcome = (
                4
                + 0.5 * treated
                + 1.0 * post
                + 2.0 * treated * post
                + 0.7 * covariate
                + rng.normal(0, 1, n_per_cell)
            )
            for index in range(n_per_cell):
                rows.append(
                    {
                        "treated_group": treated,
                        "post": post,
                        "outcome": outcome[index],
                        "covariate": covariate[index],
                    }
                )
    return pd.DataFrame(rows)


def test_panel_did_recovers_att(tmp_path) -> None:
    dataset_path = _save(tmp_path, _panel_data())

    response = run_difference_in_differences(
        dataset_path=dataset_path,
        group_column="treated_group",
        post_column="post",
        outcome="outcome",
        confounders=["baseline"],
        unit_id_column="unit_id",
    )

    assert response["success"] is True
    result = response["result"]
    assert result["estimand"] == "ATT"
    assert abs(result["ate"] - 2.5) < 0.5
    assert result["model_configuration"]["design_mode"] == (
        "two_period_panel_first_difference"
    )
    assert result["sample_size"] == 240
    assert len(result["cell_means"]) == 4


def test_repeated_cross_section_did_recovers_att(tmp_path) -> None:
    dataset_path = _save(tmp_path, _repeated_cross_section_data())

    response = run_difference_in_differences(
        dataset_path=dataset_path,
        group_column="treated_group",
        post_column="post",
        outcome="outcome",
        confounders=["covariate"],
    )

    assert response["success"] is True
    result = response["result"]
    assert abs(result["ate"] - 2.0) < 0.5
    assert result["model_configuration"]["design_mode"] == (
        "repeated_cross_section_interaction_ols"
    )
    assert result["sample_size"] == 480


def test_did_readiness_blocks_missing_four_cell_support(tmp_path) -> None:
    dataframe = _repeated_cross_section_data(n_per_cell=30)
    dataframe = dataframe[
        ~((dataframe["treated_group"] == 1) & (dataframe["post"] == 0))
    ]
    dataset_path = _save(tmp_path, dataframe)

    response = check_did_readiness(
        dataset_path=dataset_path,
        group_column="treated_group",
        post_column="post",
        outcome="outcome",
        confounders=["covariate"],
    )

    assert response["success"] is True
    assert response["result"]["ready"] is False
    assert any(
        "group_1_post_0" in issue
        for issue in response["result"]["blocking_issues"]
    )


def test_did_readiness_rejects_incomplete_panel_units(tmp_path) -> None:
    dataframe = _panel_data(unit_count=120)
    dataframe = dataframe.drop(index=dataframe.index[-1])
    dataset_path = _save(tmp_path, dataframe)

    response = check_did_readiness(
        dataset_path=dataset_path,
        group_column="treated_group",
        post_column="post",
        outcome="outcome",
        confounders=["baseline"],
        unit_id_column="unit_id",
    )

    assert response["success"] is True
    assert response["result"]["ready"] is False
    assert any(
        "同时具有 pre 和 post" in issue
        for issue in response["result"]["blocking_issues"]
    )


def test_estimator_suite_selects_did_for_explicit_design(tmp_path) -> None:
    dataset_path = _save(tmp_path, _panel_data())

    response = run_causal_estimators(
        dataset_path=dataset_path,
        treatment="treated_group",
        outcome="outcome",
        confounders=["baseline"],
        estimand="ATT",
        design_type="difference_in_differences",
        post_column="post",
        unit_id_column="unit_id",
    )

    assert response["success"] is True
    result = response["result"]
    assert result["estimand"] == "ATT"
    assert result["primary_estimator"] == "DifferenceInDifferences"
    assert abs(result["estimates"][0]["ate"] - 2.5) < 0.5
