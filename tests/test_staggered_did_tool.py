from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from causalens.tools.causal_tool import run_causal_estimators
from causalens.tools.staggered_did_tool import (
    check_staggered_did_readiness,
    run_staggered_did,
)


def _write_staggered_panel(path: Path) -> Path:
    rng = np.random.default_rng(73)
    rows: list[dict[str, float | int]] = []
    starts = [2] * 30 + [4] * 30 + [np.nan] * 30
    effects = {0: 1.0, 1: 2.0, 2: 3.0, 3: 3.0, 4: 3.0, 5: 3.0}
    for unit, start in enumerate(starts):
        unit_effect = rng.normal(scale=0.7)
        for period in range(8):
            treated = int(not np.isnan(start) and period >= start)
            event_time = period - int(start) if not np.isnan(start) else None
            effect = effects.get(event_time, 0.0) if event_time is not None else 0.0
            rows.append(
                {
                    "unit_id": unit,
                    "period": period,
                    "treatment_start": start,
                    "treated": treated,
                    "outcome": 5 + unit_effect + 0.4 * period + effect + rng.normal(scale=0.15),
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _write_confounded_staggered_panel(path: Path) -> Path:
    rng = np.random.default_rng(991)
    rows: list[dict[str, float | int]] = []
    unit_count = 180
    x_values = rng.normal(size=unit_count)
    logits = np.column_stack([0.8 * x_values, np.zeros(unit_count), -0.8 * x_values])
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    categories = [rng.choice(3, p=probabilities[index]) for index in range(unit_count)]
    starts = [2.0 if category == 0 else 4.0 if category == 1 else np.nan for category in categories]
    effects = {0: 1.0, 1: 1.5, 2: 2.0, 3: 2.0, 4: 2.0, 5: 2.0}
    for unit, (start, x_value) in enumerate(zip(starts, x_values)):
        unit_effect = rng.normal(scale=0.5)
        for period in range(8):
            treated = int(not np.isnan(start) and period >= start)
            event_time = period - int(start) if not np.isnan(start) else None
            effect = effects.get(event_time, 0.0) if event_time is not None else 0.0
            rows.append(
                {
                    "unit_id": unit,
                    "period": period,
                    "treatment_start": start,
                    "treated": treated,
                    "baseline_risk": x_value,
                    "outcome": (
                        5
                        + unit_effect
                        + 0.3 * period
                        + 0.8 * x_value * period
                        + effect
                        + rng.normal(scale=0.12)
                    ),
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_staggered_did_recovers_dynamic_effects(tmp_path: Path) -> None:
    dataset = _write_staggered_panel(tmp_path / "staggered.csv")

    response = run_staggered_did(
        dataset_path=str(dataset),
        treatment_column="treated",
        time_column="period",
        treatment_start_column="treatment_start",
        outcome="outcome",
        unit_id_column="unit_id",
        bootstrap_iterations=60,
        chart_output_dir=str(tmp_path / "charts"),
    )

    assert response["success"] is True
    result = response["result"]
    effects = {row["event_time"]: row["estimate"] for row in result["dynamic_effects"]}
    assert effects[-1] == 0.0
    assert effects[0] == pytest.approx(1.0, abs=0.2)
    assert effects[1] == pytest.approx(2.0, abs=0.2)
    assert effects[2] == pytest.approx(3.0, abs=0.2)
    assert result["group_time_effects"]
    assert result["pretrend_test"]["tested_periods"]
    assert result["placebo_test"]["pseudo_treatment_shift"] == -1
    assert len(result["cohort_sensitivity"]["leave_one_cohort_out"]) == 2
    assert Path(result["chart_path"]).is_file()


def test_staggered_readiness_rejects_inconsistent_treatment_status(
    tmp_path: Path,
) -> None:
    dataset = _write_staggered_panel(tmp_path / "invalid.csv")
    frame = pd.read_csv(dataset)
    frame.loc[0, "treated"] = 1
    frame.to_csv(dataset, index=False)

    response = check_staggered_did_readiness(
        dataset_path=str(dataset),
        treatment_column="treated",
        time_column="period",
        treatment_start_column="treatment_start",
        outcome="outcome",
        unit_id_column="unit_id",
    )

    assert response["success"] is True
    assert response["result"]["ready"] is False
    assert any(
        "处理状态必须" in issue
        for issue in response["result"]["blocking_issues"]
    )


def test_suite_selects_staggered_adoption_estimator(tmp_path: Path) -> None:
    dataset = _write_staggered_panel(tmp_path / "suite.csv")

    response = run_causal_estimators(
        dataset_path=str(dataset),
        treatment="treated",
        outcome="outcome",
        confounders=[],
        estimand="ATT",
        design_type="difference_in_differences",
        time_column="period",
        treatment_start_column="treatment_start",
        unit_id_column="unit_id",
        bootstrap_iterations=50,
    )

    assert response["success"] is True
    assert response["result"]["primary_estimator"] == "StaggeredAdoptionDID"
    assert response["result"]["estimates"][0]["group_time_effects"]
    grade = response["result"]["evidence_grade"]
    assert grade["components"]["design_family"] == "temporal"
    assert "treatment_permutation_diagnostic" not in grade["missing_checks"]


def test_doubly_robust_adjustment_uses_baseline_confounder(tmp_path: Path) -> None:
    dataset = _write_confounded_staggered_panel(tmp_path / "confounded.csv")

    response = run_staggered_did(
        dataset_path=str(dataset),
        treatment_column="treated",
        time_column="period",
        treatment_start_column="treatment_start",
        outcome="outcome",
        unit_id_column="unit_id",
        confounders=["baseline_risk"],
        cross_fitting_folds=3,
        bootstrap_iterations=30,
    )

    assert response["success"] is True
    result = response["result"]
    event_zero = next(
        row for row in result["dynamic_effects"] if row["event_time"] == 0
    )
    assert event_zero["estimate"] == pytest.approx(1.0, abs=0.25)
    assert result["model_configuration"]["adjustment"] == (
        "cross_fitted_doubly_robust"
    )
    adjusted_cells = [
        row
        for row in result["group_time_effects"]
        if row["adjustment_mode"] == "cross_fitted_doubly_robust"
    ]
    assert adjusted_cells
    assert all(row["diagnostics"]["weighted_control_ess"] > 0 for row in adjusted_cells)


def test_adjusted_overlap_threshold_can_block_execution(tmp_path: Path) -> None:
    dataset = _write_confounded_staggered_panel(tmp_path / "blocked.csv")

    response = run_staggered_did(
        dataset_path=str(dataset),
        treatment_column="treated",
        time_column="period",
        treatment_start_column="treatment_start",
        outcome="outcome",
        unit_id_column="unit_id",
        confounders=["baseline_risk"],
        cross_fitting_folds=3,
        bootstrap_iterations=30,
        min_weighted_control_ess=10_000,
    )

    assert response["success"] is False
    assert "overlap/ESS" in response["error"]
    assert response["overlap_diagnostics"]["min_weighted_control_ess"] < 10_000
