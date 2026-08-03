import numpy as np
import pandas as pd

from causalens.tools.causal_tool import (
    run_dml,
)


def save_test_data(
    tmp_path,
    dataframe: pd.DataFrame,
) -> str:
    dataset_path = (
        tmp_path / "dml_data.csv"
    )

    dataframe.to_csv(
        dataset_path,
        index=False,
    )

    return str(dataset_path)


def build_causal_data(
    n: int = 1200,
) -> pd.DataFrame:
    rng = np.random.default_rng(42)

    age = rng.normal(35, 8, n)
    log_income = rng.normal(
        10.5,
        0.4,
        n,
    )
    activity = rng.normal(
        50,
        12,
        n,
    )
    channel = rng.normal(
        0,
        1,
        n,
    )

    treatment_logit = (
        -1.0
        + 0.02 * (age - 35)
        + 0.60 * (
            log_income - 10.5
        )
        + 0.03 * (
            activity - 50
        )
        + 0.50 * channel
    )

    treatment_probability = (
        1
        / (
            1
            + np.exp(
                -treatment_logit
            )
        )
    )

    treatment = rng.binomial(
        1,
        treatment_probability,
    )

    outcome = (
        20
        + 3.0 * treatment
        + 0.08 * age
        + 2.0 * (
            log_income - 10.5
        )
        + 0.12 * activity
        + 1.5 * channel
        + rng.normal(
            0,
            4,
            n,
        )
    )

    return pd.DataFrame(
        {
            "treatment": treatment,
            "outcome": outcome,
            "age": age,
            "log_income": log_income,
            "activity": activity,
            "channel": channel,
        }
    )


def test_run_dml_returns_ate_and_interval(
    tmp_path,
):
    dataset_path = save_test_data(
        tmp_path,
        build_causal_data(),
    )

    response = run_dml(
        dataset_path=dataset_path,
        treatment="treatment",
        outcome="outcome",
        confounders=[
            "age",
            "log_income",
            "activity",
            "channel",
        ],
        random_state=42,
        cv=5,
    )

    assert response["success"] is True

    result = response["result"]

    assert (
        result["analysis_type"]
        == "linear_dml"
    )
    assert result["sample_size"] == 1200
    assert abs(
        result["ate"] - 3.0
    ) < 1.0

    lower_bound, upper_bound = (
        result["confidence_interval"]
    )

    assert (
        lower_bound
        < result["ate"]
        < upper_bound
    )

    assert (
        result["naive_group_difference"]
        is not None
    )
    assert len(
        result["assumptions"]
    ) > 0
    assert len(
        result["warnings"]
    ) > 0


def test_run_dml_is_blocked_when_not_ready(
    tmp_path,
):
    dataset_path = save_test_data(
        tmp_path,
        build_causal_data(
            n=40
        ),
    )

    response = run_dml(
        dataset_path=dataset_path,
        treatment="treatment",
        outcome="outcome",
        confounders=["age"],
    )

    assert response["success"] is False
    assert "not executed" in (
        response["error"]
    )
    assert response["readiness"][
        "ready"
    ] is False


def test_run_dml_rejects_continuous_mode(
    tmp_path,
):
    dataset_path = save_test_data(
        tmp_path,
        build_causal_data(),
    )

    response = run_dml(
        dataset_path=dataset_path,
        treatment="treatment",
        outcome="outcome",
        confounders=["age"],
        discrete_treatment=False,
    )

    assert response["success"] is False
    assert "binary discrete" in (
        response["error"]
    )


def test_run_dml_rejects_invalid_cv(
    tmp_path,
):
    dataset_path = save_test_data(
        tmp_path,
        build_causal_data(),
    )

    response = run_dml(
        dataset_path=dataset_path,
        treatment="treatment",
        outcome="outcome",
        confounders=["age"],
        cv=1,
    )

    assert response["success"] is False
    assert "at least 2" in (
        response["error"]
    )