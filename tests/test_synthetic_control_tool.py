from __future__ import annotations

import numpy as np
import pandas as pd

from causalens.tools.synthetic_control_tool import (
    check_synthetic_control_readiness,
    run_synthetic_control,
)
from causalens.tools.causal_tool import run_causal_estimators


def _panel(*, effect: float = 3.0, donor_count: int = 8) -> pd.DataFrame:
    rng = np.random.default_rng(20261014)
    periods = range(12)
    donor_intercepts = rng.normal(scale=0.4, size=donor_count)
    rows: list[dict[str, float | int | str]] = []
    donor_paths: list[np.ndarray] = []
    for donor in range(donor_count):
        path = np.asarray(
            [
                5
                + donor_intercepts[donor]
                + 0.25 * period
                + 0.15 * np.sin(period)
                + rng.normal(scale=0.03)
                for period in periods
            ]
        )
        donor_paths.append(path)
        for period, outcome in zip(periods, path):
            rows.append(
                {
                    "unit": f"donor_{donor}",
                    "period": period,
                    "treated": 0,
                    "outcome": outcome,
                    "size": 10 + donor,
                }
            )
    weights = np.asarray([0.55, 0.30, 0.15, *([0.0] * (donor_count - 3))])
    untreated = np.average(np.vstack(donor_paths), axis=0, weights=weights)
    for period, baseline in zip(periods, untreated):
        rows.append(
            {
                "unit": "treated",
                "period": period,
                "treated": int(period >= 8),
                "outcome": baseline + (effect if period >= 8 else 0.0),
                "size": float(np.average(np.arange(10, 10 + donor_count), weights=weights)),
            }
        )
    return pd.DataFrame(rows)


def test_synthetic_control_recovers_effect_and_simplex_weights(tmp_path):
    path = tmp_path / "panel.csv"
    _panel().to_csv(path, index=False)

    response = run_synthetic_control(
        dataset_path=str(path),
        treatment="treated",
        outcome="outcome",
        unit_id_column="unit",
        time_column="period",
        intervention_time=8,
        covariates=["size"],
    )

    assert response["success"] is True
    result = response["result"]
    assert result["estimator_name"] == "SyntheticControl"
    assert abs(result["ate"] - 3.0) < 0.15
    assert abs(sum(row["weight"] for row in result["donor_weights"]) - 1) < 1e-6
    assert result["fit_diagnostics"]["pre_rmspe"] < 0.10
    assert result["placebo_inference"]["placebo_count"] == 8


def test_synthetic_control_readiness_blocks_short_pre_period(tmp_path):
    frame = _panel()
    frame = frame[frame["period"] >= 4]
    path = tmp_path / "short.csv"
    frame.to_csv(path, index=False)

    response = check_synthetic_control_readiness(
        dataset_path=str(path),
        treatment="treated",
        outcome="outcome",
        unit_id_column="unit",
        time_column="period",
        intervention_time=8,
    )

    assert response["success"] is True
    assert response["result"]["ready"] is False
    assert any("five pre periods" in issue for issue in response["result"]["blocking_issues"])


def test_synthetic_control_blocks_multiple_treated_units(tmp_path):
    frame = _panel()
    mask = (frame["unit"] == "donor_0") & (frame["period"] >= 8)
    frame.loc[mask, "treated"] = 1
    path = tmp_path / "multiple.csv"
    frame.to_csv(path, index=False)

    response = check_synthetic_control_readiness(
        dataset_path=str(path),
        treatment="treated",
        outcome="outcome",
        unit_id_column="unit",
        time_column="period",
        intervention_time=8,
    )

    assert response["success"] is False
    assert "exactly one treated unit" in response["error"]


def test_synthetic_control_rejects_duplicate_unit_time_rows(tmp_path):
    frame = _panel()
    frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    path = tmp_path / "duplicates.csv"
    frame.to_csv(path, index=False)

    response = check_synthetic_control_readiness(
        dataset_path=str(path),
        treatment="treated",
        outcome="outcome",
        unit_id_column="unit",
        time_column="period",
        intervention_time=8,
    )

    assert response["success"] is False
    assert "one row per unit-time pair" in response["error"]


def test_synthetic_control_runs_through_causal_suite(tmp_path):
    path = tmp_path / "suite.csv"
    _panel().to_csv(path, index=False)

    response = run_causal_estimators(
        dataset_path=str(path),
        treatment="treated",
        outcome="outcome",
        confounders=["size"],
        estimand="ATT",
        design_type="synthetic_control",
        unit_id_column="unit",
        time_column="period",
        intervention_time=8,
        permutation_iterations=10,
    )

    assert response["success"] is True
    result = response["result"]
    assert result["primary_estimator"] == "SyntheticControl"
    assert result["design_selection"]["study_design"] == (
        "synthetic_control_single_treated_unit"
    )
    assert result["data_quality"]["ready"] is True
    assert result["permutation_placebo"]["available"] is False


def test_causal_suite_reuses_content_addressed_result_cache(tmp_path):
    path = tmp_path / "cached.csv"
    _panel().to_csv(path, index=False)
    arguments = {
        "dataset_path": str(path),
        "treatment": "treated",
        "outcome": "outcome",
        "confounders": [],
        "estimand": "ATT",
        "design_type": "synthetic_control",
        "unit_id_column": "unit",
        "time_column": "period",
        "intervention_time": 8,
        "result_cache_dir": str(tmp_path / "cache"),
    }

    first = run_causal_estimators(**arguments)
    second = run_causal_estimators(**arguments)

    assert first["result"]["runtime"]["cache_hit"] is False
    assert second["result"]["runtime"]["cache_hit"] is True
    assert second["result"]["estimates"] == first["result"]["estimates"]
