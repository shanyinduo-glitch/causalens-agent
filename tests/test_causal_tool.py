import numpy as np
import pandas as pd

from causalens.tools import causal_tool
from causalens.tools.causal_tool import (
    run_aipw,
    run_causal_forest,
    run_causal_estimators,
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


def build_heterogeneous_causal_data(n: int = 1400) -> pd.DataFrame:
    rng = np.random.default_rng(123)
    age = rng.normal(40, 9, n)
    baseline = rng.normal(0, 1, n)
    treatment_probability = 1 / (
        1 + np.exp(-(-0.3 + 0.025 * (age - 40) + 0.5 * baseline))
    )
    treatment = rng.binomial(1, treatment_probability)
    true_effect = 2.0 + 0.14 * (age - 40)
    outcome = (
        10
        + 1.2 * baseline
        + 0.08 * age
        + true_effect * treatment
        + rng.normal(0, 2, n)
    )
    return pd.DataFrame(
        {
            "treatment": treatment,
            "outcome": outcome,
            "age": age,
            "baseline": baseline,
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


def test_run_aipw_returns_doubly_robust_ate(tmp_path):
    dataset_path = save_test_data(tmp_path, build_causal_data())

    response = run_aipw(
        dataset_path=dataset_path,
        treatment="treatment",
        outcome="outcome",
        confounders=["age", "log_income", "activity", "channel"],
        random_state=42,
        cv=5,
    )

    assert response["success"] is True
    result = response["result"]
    assert result["analysis_type"] == "aipw"
    assert result["estimator_name"] == "AIPW"
    assert abs(result["ate"] - 3.0) < 1.0
    assert result["standard_error"] > 0
    lower, upper = result["confidence_interval"]
    assert lower < result["ate"] < upper
    assert result["model_configuration"]["cross_fitting_folds"] == 5


def test_run_causal_estimators_compares_aipw_and_dml(tmp_path):
    dataset_path = save_test_data(tmp_path, build_causal_data())

    response = run_causal_estimators(
        dataset_path=dataset_path,
        treatment="treatment",
        outcome="outcome",
        confounders=["age", "log_income", "activity", "channel"],
        estimand="ATE",
        random_state=42,
        cv=5,
    )

    assert response["success"] is True
    result = response["result"]
    assert result["analysis_type"] == "causal_estimation_suite"
    assert result["primary_estimator"] == "AIPW"
    assert result["robustness_estimators"] == ["LinearDML"]
    assert [item["estimator"] for item in result["estimates"]] == [
        "AIPW",
        "LinearDML",
    ]
    assert result["comparison"]["estimate_count"] == 2
    assert result["comparison"]["absolute_ate_difference"] >= 0
    assert result["sensitivity_summary"]["method"] == (
        "cinelli_hazlett_partial_r2"
    )
    assert result["sensitivity_summary"]["e_value_computed"] is False
    placebo = result["permutation_placebo"]
    assert placebo["available"] is True
    assert placebo["signal_status"] == "stronger_than_permuted_labels"
    assert placebo["iterations"] == 200
    evidence_grade = result["evidence_grade"]
    assert evidence_grade["grade"] == "moderate_support"
    assert evidence_grade["label_zh"] == "中等支持"
    assert "negative_control_outcome" in evidence_grade["missing_checks"]
    for estimate in result["estimates"]:
        sensitivity = estimate["sensitivity_analysis"]
        assert 0 <= sensitivity["robustness_value_point_estimate"] <= 1
        assert 0 <= sensitivity["robustness_value_significance"] <= 1
        benchmark = sensitivity["observed_confounder_benchmark"]
        assert benchmark["available"] is True
        assert len(benchmark["confounders"]) == 4
    primary_benchmark = result["sensitivity_summary"][
        "observed_confounder_benchmark"
    ]
    assert primary_benchmark["strongest_confounder"] in {
        "age",
        "log_income",
        "activity",
        "channel",
    }


def test_causal_suite_runs_declared_negative_control_outcome(
    tmp_path,
    monkeypatch,
):
    dataframe = build_causal_data(n=200)
    rng = np.random.default_rng(2026)
    dataframe["pre_treatment_score"] = (
        0.2 * dataframe["age"]
        + 0.5 * dataframe["activity"]
        + rng.normal(0, 3, len(dataframe))
    )
    dataset_path = save_test_data(tmp_path, dataframe)

    def fake_result(outcome: str, *, include_standard_error: bool) -> dict:
        is_negative_control = outcome == "pre_treatment_score"
        estimate = 0.03 if is_negative_control else 3.0
        interval = [-0.10, 0.16] if is_negative_control else [2.6, 3.4]
        result = {
            "analysis_type": "aipw",
            "ate": estimate,
            "confidence_interval": interval,
            "confidence_level": 0.95,
            "sample_size": len(dataframe),
            "warnings": [],
            "assumptions": [],
        }
        if include_standard_error:
            result["standard_error"] = 0.07 if is_negative_control else 0.2
        return {"success": True, "result": result}

    monkeypatch.setattr(
        causal_tool,
        "run_aipw",
        lambda **kwargs: fake_result(
            kwargs["outcome"], include_standard_error=True
        ),
    )
    monkeypatch.setattr(
        causal_tool,
        "run_dml",
        lambda **kwargs: fake_result(
            kwargs["outcome"], include_standard_error=False
        ),
    )

    response = causal_tool.run_causal_estimators(
        dataset_path=dataset_path,
        treatment="treatment",
        outcome="outcome",
        confounders=["age", "log_income", "activity", "channel"],
        negative_control_outcomes=["pre_treatment_score"],
        random_state=42,
        cv=5,
    )

    assert response["success"] is True
    summary = response["result"]["negative_control_summary"]
    assert summary["requested_outcomes"] == ["pre_treatment_score"]
    assert summary["result_count"] == 1
    assert summary["results"][0]["success"] is True
    assert summary["results"][0]["passed"] is True
    assert "all_declared_negative_controls_passed" in (
        response["result"]["evidence_grade"]["supporting_signals"]
    )


def test_run_causal_estimators_rejects_unsupported_estimand(tmp_path):
    dataset_path = save_test_data(tmp_path, build_causal_data())

    response = run_causal_estimators(
        dataset_path=dataset_path,
        treatment="treatment",
        outcome="outcome",
        confounders=["age"],
        estimand="ATT",
    )

    assert response["success"] is False
    assert "No registered estimator" in response["error"]


def test_run_causal_forest_recovers_direction_of_heterogeneity(tmp_path):
    dataset_path = save_test_data(
        tmp_path,
        build_heterogeneous_causal_data(),
    )

    response = run_causal_forest(
        dataset_path=dataset_path,
        treatment="treatment",
        outcome="outcome",
        confounders=["age", "baseline"],
        effect_modifiers=["age"],
        random_state=42,
        cv=3,
    )

    assert response["success"] is True
    result = response["result"]
    assert result["analysis_type"] == "causal_forest_dml"
    assert abs(result["ate"] - 2.0) < 1.0
    assert result["heterogeneity"]["cate_p90"] > result["heterogeneity"][
        "cate_p10"
    ]
    subgroup = result["heterogeneity"]["median_split_subgroups"][0]
    assert subgroup["effect_modifier"] == "age"
    assert subgroup["upper_minus_lower_cate"] > 0.3


def test_cate_suite_selects_causal_forest(tmp_path):
    dataset_path = save_test_data(
        tmp_path,
        build_heterogeneous_causal_data(),
    )

    response = run_causal_estimators(
        dataset_path=dataset_path,
        treatment="treatment",
        outcome="outcome",
        confounders=["age", "baseline"],
        effect_modifiers=["age"],
        estimand="CATE",
        random_state=42,
        cv=3,
    )

    assert response["success"] is True
    result = response["result"]
    assert result["estimand"] == "CATE"
    assert result["primary_estimator"] == "CausalForestDML"
    assert result["robustness_estimators"] == []
    assert result["estimates"][0]["effect_modifiers"] == ["age"]


def test_cate_suite_requires_effect_modifier(tmp_path):
    dataset_path = save_test_data(tmp_path, build_causal_data())

    response = run_causal_estimators(
        dataset_path=dataset_path,
        treatment="treatment",
        outcome="outcome",
        confounders=["age"],
        estimand="CATE",
    )

    assert response["success"] is False
    assert "effect modifier" in response["error"]
