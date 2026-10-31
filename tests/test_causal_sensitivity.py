from __future__ import annotations

import pytest
import numpy as np
import pandas as pd

from causalens.causal.sensitivity import (
    build_observed_confounder_benchmark,
    build_sensitivity_analysis,
    robustness_value,
)


def test_robustness_value_matches_published_reference_example() -> None:
    point = robustness_value(t_statistic=4.18, degrees_of_freedom=783)
    significance = robustness_value(
        t_statistic=4.18,
        degrees_of_freedom=783,
        alpha=0.05,
    )

    assert point == pytest.approx(0.139, abs=0.001)
    assert significance == pytest.approx(0.076, abs=0.001)


def test_sensitivity_analysis_infers_standard_error_and_explains_e_value() -> None:
    result = build_sensitivity_analysis(
        estimate=2.0,
        standard_error=None,
        confidence_interval=[1.0, 3.0],
        sample_size=200,
        adjusted_covariate_count=4,
    )

    assert result["standard_error_inferred_from_interval"] is True
    assert 0 <= result["robustness_value_significance"] <= 1
    assert result["e_value"]["computed"] is False
    assert "risk-ratio" in result["e_value"]["reason"]


def test_sensitivity_analysis_marks_interval_crossing_zero_inconclusive() -> None:
    result = build_sensitivity_analysis(
        estimate=0.1,
        standard_error=0.2,
        confidence_interval=[-0.3, 0.5],
        sample_size=300,
        adjusted_covariate_count=2,
    )

    assert result["robustness_value_significance"] == 0
    assert result["unmeasured_confounding_risk"] == "statistically_inconclusive"


def test_observed_confounder_benchmark_ranks_stronger_covariate_first() -> None:
    rng = np.random.default_rng(42)
    strong = rng.normal(size=800)
    weak = rng.normal(size=800)
    treatment = (strong + rng.normal(scale=0.7, size=800) > 0).astype(int)
    outcome = (
        2.0 * treatment
        + 2.5 * strong
        + 0.1 * weak
        + rng.normal(size=800)
    )
    dataframe = pd.DataFrame(
        {
            "treatment": treatment,
            "outcome": outcome,
            "strong": strong,
            "weak": weak,
        }
    )

    result = build_observed_confounder_benchmark(
        dataframe=dataframe,
        treatment="treatment",
        outcome="outcome",
        confounders=["strong", "weak"],
        robustness_value_point_estimate=0.20,
        robustness_value_significance=0.10,
    )

    assert result["available"] is True
    assert result["strongest_confounder"] == "strong"
    assert result["confounders"][0]["equivalent_robustness_value"] > (
        result["confounders"][1]["equivalent_robustness_value"]
    )
    assert result["strongest_exceeds_significance_rv"] is True
