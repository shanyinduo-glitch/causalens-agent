import numpy as np
import pandas as pd

from causalens.tools.causal_readiness import (
    check_causal_readiness,
)


def save_test_data(
    tmp_path,
    dataframe: pd.DataFrame,
) -> str:
    dataset_path = (
        tmp_path / "causal_data.csv"
    )

    dataframe.to_csv(
        dataset_path,
        index=False,
    )

    return str(dataset_path)


def build_valid_causal_data(
    n: int = 300,
) -> pd.DataFrame:
    rng = np.random.default_rng(42)

    age = rng.normal(35, 8, n)
    activity = rng.normal(50, 10, n)
    treatment = rng.binomial(
        1,
        0.45,
        n,
    )
    outcome = (
        3 * treatment
        + 0.1 * age
        + 0.2 * activity
        + rng.normal(0, 2, n)
    )

    return pd.DataFrame(
        {
            "treatment": treatment,
            "outcome": outcome,
            "age": age,
            "activity": activity,
        }
    )


def test_readiness_passes_valid_data(
    tmp_path,
):
    dataset_path = save_test_data(
        tmp_path,
        build_valid_causal_data(),
    )

    response = check_causal_readiness(
        dataset_path=dataset_path,
        treatment="treatment",
        outcome="outcome",
        confounders=[
            "age",
            "activity",
        ],
    )

    assert response["success"] is True

    result = response["result"]

    assert result["ready"] is True
    assert result["blocking_issues"] == []
    assert result["sample_summary"][
        "complete_cases"
    ] == 300


def test_missing_treatment_is_blocked(
    tmp_path,
):
    dataset_path = save_test_data(
        tmp_path,
        build_valid_causal_data(),
    )

    response = check_causal_readiness(
        dataset_path=dataset_path,
        treatment="unknown_treatment",
        outcome="outcome",
        confounders=["age"],
    )

    assert response["success"] is True
    assert response["result"]["ready"] is False
    assert "do not exist" in " ".join(
        response["result"][
            "blocking_issues"
        ]
    )


def test_missing_outcome_is_blocked(
    tmp_path,
):
    dataset_path = save_test_data(
        tmp_path,
        build_valid_causal_data(),
    )

    response = check_causal_readiness(
        dataset_path=dataset_path,
        treatment="treatment",
        outcome="unknown_outcome",
        confounders=["age"],
    )

    assert response["result"]["ready"] is False


def test_no_confounders_is_blocked(
    tmp_path,
):
    dataset_path = save_test_data(
        tmp_path,
        build_valid_causal_data(),
    )

    response = check_causal_readiness(
        dataset_path=dataset_path,
        treatment="treatment",
        outcome="outcome",
        confounders=[],
    )

    assert response["success"] is True
    assert response["result"]["ready"] is False
    assert "At least one confounder" in (
        " ".join(
            response["result"][
                "blocking_issues"
            ]
        )
    )


def test_constant_treatment_is_blocked(
    tmp_path,
):
    dataframe = build_valid_causal_data()
    dataframe["treatment"] = 1

    dataset_path = save_test_data(
        tmp_path,
        dataframe,
    )

    response = check_causal_readiness(
        dataset_path=dataset_path,
        treatment="treatment",
        outcome="outcome",
        confounders=["age"],
    )

    assert response["result"]["ready"] is False
    assert "has no variation" in " ".join(
        response["result"][
            "blocking_issues"
        ]
    )


def test_constant_outcome_is_blocked(
    tmp_path,
):
    dataframe = build_valid_causal_data()
    dataframe["outcome"] = 10

    dataset_path = save_test_data(
        tmp_path,
        dataframe,
    )

    response = check_causal_readiness(
        dataset_path=dataset_path,
        treatment="treatment",
        outcome="outcome",
        confounders=["age"],
    )

    assert response["result"]["ready"] is False
    assert "Outcome" in " ".join(
        response["result"][
            "blocking_issues"
        ]
    )


def test_constant_confounder_is_blocked(
    tmp_path,
):
    dataframe = build_valid_causal_data()
    dataframe["age"] = 35

    dataset_path = save_test_data(
        tmp_path,
        dataframe,
    )

    response = check_causal_readiness(
        dataset_path=dataset_path,
        treatment="treatment",
        outcome="outcome",
        confounders=[
            "age",
            "activity",
        ],
    )

    assert response["result"]["ready"] is False
    assert "confounders have no variation" in (
        " ".join(
            response["result"][
                "blocking_issues"
            ]
        )
    )


def test_small_sample_is_blocked(
    tmp_path,
):
    dataset_path = save_test_data(
        tmp_path,
        build_valid_causal_data(
            n=40
        ),
    )

    response = check_causal_readiness(
        dataset_path=dataset_path,
        treatment="treatment",
        outcome="outcome",
        confounders=["age"],
    )

    assert response["result"]["ready"] is False
    assert "Too few complete observations" in (
        " ".join(
            response["result"][
                "blocking_issues"
            ]
        )
    )


def test_high_missing_rate_is_blocked(
    tmp_path,
):
    dataframe = build_valid_causal_data()

    dataframe.loc[
        :100,
        "age",
    ] = np.nan

    dataset_path = save_test_data(
        tmp_path,
        dataframe,
    )

    response = check_causal_readiness(
        dataset_path=dataset_path,
        treatment="treatment",
        outcome="outcome",
        confounders=["age"],
    )

    assert response["result"]["ready"] is False
    assert "maximum missing rate" in (
        " ".join(
            response["result"][
                "blocking_issues"
            ]
        )
    )


def test_duplicate_confounders_are_blocked(
    tmp_path,
):
    dataset_path = save_test_data(
        tmp_path,
        build_valid_causal_data(),
    )

    response = check_causal_readiness(
        dataset_path=dataset_path,
        treatment="treatment",
        outcome="outcome",
        confounders=[
            "age",
            "age",
        ],
    )

    assert response["result"]["ready"] is False
    assert "duplicate variables" in " ".join(
        response["result"][
            "blocking_issues"
        ]
    )


def test_treatment_cannot_be_confounder(
    tmp_path,
):
    dataset_path = save_test_data(
        tmp_path,
        build_valid_causal_data(),
    )

    response = check_causal_readiness(
        dataset_path=dataset_path,
        treatment="treatment",
        outcome="outcome",
        confounders=[
            "treatment",
            "age",
        ],
    )

    assert response["result"]["ready"] is False
    assert "Treatment cannot also" in " ".join(
        response["result"][
            "blocking_issues"
        ]
    )


def test_continuous_treatment_is_blocked(
    tmp_path,
):
    dataframe = build_valid_causal_data()
    dataframe["treatment"] = np.arange(
        len(dataframe)
    )

    dataset_path = save_test_data(
        tmp_path,
        dataframe,
    )

    response = check_causal_readiness(
        dataset_path=dataset_path,
        treatment="treatment",
        outcome="outcome",
        confounders=["age"],
    )

    assert response["result"]["ready"] is False
    assert "supports binary treatment only" in (
        " ".join(
            response["result"][
                "blocking_issues"
            ]
        )
    )

def test_non_zero_one_treatment_is_blocked(
    tmp_path,
):
    dataframe = build_valid_causal_data()

    dataframe["treatment"] = (
        dataframe["treatment"]
        .map(
            {
                0: 1,
                1: 2,
            }
        )
    )

    dataset_path = save_test_data(
        tmp_path,
        dataframe,
    )

    response = check_causal_readiness(
        dataset_path=dataset_path,
        treatment="treatment",
        outcome="outcome",
        confounders=[
            "age",
            "activity",
        ],
    )

    assert response["success"] is True
    assert response["result"]["ready"] is False

    blocking_text = " ".join(
        response["result"][
            "blocking_issues"
        ]
    )

    assert "coded as 0 and 1" in (
        blocking_text
    )