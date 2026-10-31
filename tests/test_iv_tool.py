from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from causalens.tools.iv_tool import (
    check_iv_readiness,
    run_two_stage_least_squares,
)
from causalens.tools.causal_tool import run_causal_estimators


def _write_iv_data(
    path: Path,
    *,
    instrument_strength: float = 0.9,
    n: int = 1200,
) -> Path:
    rng = np.random.default_rng(2026)
    x = rng.normal(size=n)
    z1 = rng.normal(size=n)
    z2 = rng.normal(size=n)
    latent = rng.normal(size=n)
    treatment = (
        instrument_strength * z1
        + 0.6 * instrument_strength * z2
        + 0.5 * x
        + latent
        + rng.normal(scale=0.3, size=n)
    )
    outcome = (
        2.0 * treatment
        + 0.7 * x
        + 0.8 * latent
        + rng.normal(size=n)
    )
    pd.DataFrame(
        {
            "z1": z1,
            "z2": z2,
            "x": x,
            "treatment": treatment,
            "outcome": outcome,
        }
    ).to_csv(path, index=False)
    return path


def test_iv_readiness_reports_strong_first_stage(tmp_path: Path) -> None:
    dataset = _write_iv_data(tmp_path / "iv.csv")

    response = check_iv_readiness(
        dataset_path=str(dataset),
        treatment="treatment",
        outcome="outcome",
        instruments=["z1", "z2"],
        confounders=["x"],
    )

    assert response["success"] is True
    result = response["result"]
    assert result["ready"] is True
    assert result["first_stage"]["f_statistic"] > 10
    assert result["first_stage"]["partial_r2"] > 0.2
    assert result["first_stage"]["weak_instrument_warning"] is False


def test_two_stage_least_squares_recovers_effect(tmp_path: Path) -> None:
    dataset = _write_iv_data(tmp_path / "iv.csv")

    response = run_two_stage_least_squares(
        dataset_path=str(dataset),
        treatment="treatment",
        outcome="outcome",
        instruments=["z1", "z2"],
        confounders=["x"],
    )

    assert response["success"] is True
    result = response["result"]
    assert result["estimand"] == "LATE"
    assert result["ate"] == pytest.approx(2.0, abs=0.15)
    assert result["confidence_interval"][0] < 2.0
    assert result["confidence_interval"][1] > 2.0
    assert result["overidentification_test"]["available"] is True
    assert result["endogeneity_test"]["rejects_exogeneity_at_0_05"] is True


def test_iv_readiness_warns_for_weak_instruments(tmp_path: Path) -> None:
    dataset = _write_iv_data(
        tmp_path / "weak.csv",
        instrument_strength=0.01,
        n=600,
    )

    response = check_iv_readiness(
        dataset_path=str(dataset),
        treatment="treatment",
        outcome="outcome",
        instruments=["z1", "z2"],
        confounders=["x"],
    )

    assert response["success"] is True
    assert response["result"]["ready"] is True
    assert response["result"]["first_stage"]["weak_instrument_warning"] is True
    assert any("weak" in warning.lower() for warning in response["result"]["warnings"])


def test_iv_rejects_instrument_role_overlap(tmp_path: Path) -> None:
    dataset = _write_iv_data(tmp_path / "overlap.csv")

    response = check_iv_readiness(
        dataset_path=str(dataset),
        treatment="treatment",
        outcome="outcome",
        instruments=["x"],
        confounders=["x"],
    )

    assert response["success"] is False
    assert "distinct" in response["error"]


def test_causal_suite_selects_iv_and_builds_evidence_grade(
    tmp_path: Path,
) -> None:
    dataset = _write_iv_data(tmp_path / "suite.csv")

    response = run_causal_estimators(
        dataset_path=str(dataset),
        treatment="treatment",
        outcome="outcome",
        confounders=["x"],
        instruments=["z1", "z2"],
        estimand="LATE",
        design_type="instrumental_variable",
    )

    assert response["success"] is True
    result = response["result"]
    assert result["primary_estimator"] == "TwoStageLeastSquares"
    assert result["instruments"] == ["z1", "z2"]
    assert result["estimates"][0]["first_stage"]["f_statistic"] > 10
    grade = result["evidence_grade"]
    assert grade["components"]["design_family"] == "instrumental_variable"
    assert "iv_first_stage_above_threshold" in grade["supporting_signals"]
    assert "treatment_permutation_diagnostic" not in grade["missing_checks"]
