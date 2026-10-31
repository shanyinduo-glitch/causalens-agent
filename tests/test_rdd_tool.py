from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from causalens.tools.rdd_tool import (
    check_rdd_readiness,
    run_sharp_regression_discontinuity,
)
from causalens.tools.causal_tool import run_causal_estimators


def _write_rdd_data(path: Path, *, n: int = 2400) -> Path:
    rng = np.random.default_rng(2027)
    running = rng.uniform(-2.0, 2.0, size=n)
    treatment = (running >= 0).astype(int)
    baseline = rng.normal(size=n)
    outcome = (
        1.0
        + 0.6 * running
        + 2.0 * treatment
        + 0.2 * treatment * running
        + 0.5 * baseline
        + rng.normal(scale=0.8, size=n)
    )
    pd.DataFrame(
        {
            "running": running,
            "treatment": treatment,
            "baseline": baseline,
            "outcome": outcome,
        }
    ).to_csv(path, index=False)
    return path


def test_rdd_readiness_checks_cutoff_support_and_assignment(tmp_path: Path) -> None:
    dataset = _write_rdd_data(tmp_path / "rdd.csv")

    response = check_rdd_readiness(
        dataset_path=str(dataset),
        treatment="treatment",
        outcome="outcome",
        running_variable="running",
        cutoff=0.0,
        covariates=["baseline"],
    )

    assert response["success"] is True
    result = response["result"]
    assert result["ready"] is True
    assert result["bandwidth"] > 0
    assert result["sample_summary"]["assignment_compliance"] == 1.0
    assert result["sample_summary"]["left_count_within_bandwidth"] >= 30
    assert result["sample_summary"]["right_count_within_bandwidth"] >= 30


def test_sharp_rdd_recovers_local_effect_and_bandwidth_stability(
    tmp_path: Path,
) -> None:
    dataset = _write_rdd_data(tmp_path / "rdd.csv")

    response = run_sharp_regression_discontinuity(
        dataset_path=str(dataset),
        treatment="treatment",
        outcome="outcome",
        running_variable="running",
        cutoff=0.0,
        covariates=["baseline"],
    )

    assert response["success"] is True
    result = response["result"]
    assert result["estimand"] == "LATE"
    assert result["ate"] == pytest.approx(2.0, abs=0.3)
    assert result["confidence_interval"][0] < 2.0
    assert result["confidence_interval"][1] > 2.0
    assert result["bandwidth_sensitivity"]["available_count"] == 5
    assert result["bandwidth_sensitivity"]["direction_stable"] is True
    assert result["donut_sensitivity"]["available"] is True
    assert result["covariate_continuity"]["available"] is True


def test_sharp_rdd_blocks_noncompliant_fuzzy_assignment(tmp_path: Path) -> None:
    dataset = _write_rdd_data(tmp_path / "fuzzy.csv", n=800)
    frame = pd.read_csv(dataset)
    frame.loc[:99, "treatment"] = 1 - frame.loc[:99, "treatment"]
    frame.to_csv(dataset, index=False)

    response = check_rdd_readiness(
        dataset_path=str(dataset),
        treatment="treatment",
        outcome="outcome",
        running_variable="running",
        cutoff=0.0,
        covariates=["baseline"],
    )

    assert response["success"] is True
    assert response["result"]["ready"] is False
    assert any(
        "fuzzy RDD" in issue
        for issue in response["result"]["blocking_issues"]
    )


def test_rdd_rejects_role_overlap(tmp_path: Path) -> None:
    dataset = _write_rdd_data(tmp_path / "overlap.csv")

    response = check_rdd_readiness(
        dataset_path=str(dataset),
        treatment="treatment",
        outcome="outcome",
        running_variable="running",
        cutoff=0.0,
        covariates=["running"],
    )

    assert response["success"] is False
    assert "distinct" in response["error"]


def test_causal_suite_selects_sharp_rdd_and_builds_grade(
    tmp_path: Path,
) -> None:
    dataset = _write_rdd_data(tmp_path / "suite.csv")

    response = run_causal_estimators(
        dataset_path=str(dataset),
        treatment="treatment",
        outcome="outcome",
        confounders=["baseline"],
        estimand="LATE",
        design_type="regression_discontinuity",
        running_variable="running",
        cutoff=0.0,
    )

    assert response["success"] is True
    result = response["result"]
    assert result["primary_estimator"] == "SharpRegressionDiscontinuity"
    assert result["running_variable"] == "running"
    assert result["cutoff"] == 0.0
    assert result["bandwidth"] > 0
    grade = result["evidence_grade"]
    assert grade["components"]["design_family"] == (
        "regression_discontinuity"
    )
    assert "rdd_bandwidth_direction_stable" in grade["supporting_signals"]
    assert "treatment_permutation_diagnostic" not in grade["missing_checks"]
