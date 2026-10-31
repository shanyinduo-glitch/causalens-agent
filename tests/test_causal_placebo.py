from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from causalens.causal.placebo import run_residualized_treatment_permutation


def test_residualized_permutation_detects_strong_treatment_signal() -> None:
    rng = np.random.default_rng(7)
    sample_size = 600
    x1 = rng.normal(size=sample_size)
    x2 = rng.normal(size=sample_size)
    propensity = 1 / (1 + np.exp(-0.7 * x1 + 0.4 * x2))
    treatment = rng.binomial(1, propensity)
    outcome = 2.5 * treatment + x1 - x2 + rng.normal(size=sample_size)
    dataframe = pd.DataFrame(
        {
            "treatment": treatment,
            "outcome": outcome,
            "x1": x1,
            "x2": x2,
        }
    )

    result = run_residualized_treatment_permutation(
        dataframe=dataframe,
        treatment="treatment",
        outcome="outcome",
        confounders=["x1", "x2"],
        iterations=99,
        random_state=11,
    )

    assert result["available"] is True
    assert result["signal_status"] == "stronger_than_permuted_labels"
    assert result["empirical_two_sided_p_value"] == pytest.approx(0.01)
    assert abs(result["placebo_mean"]) < 0.1


def test_residualized_permutation_requires_a_minimum_budget() -> None:
    dataframe = pd.DataFrame(
        {
            "treatment": [0, 1] * 20,
            "outcome": np.arange(40),
            "x": np.arange(40),
        }
    )

    with pytest.raises(ValueError, match="at least 20"):
        run_residualized_treatment_permutation(
            dataframe=dataframe,
            treatment="treatment",
            outcome="outcome",
            confounders=["x"],
            iterations=10,
        )
