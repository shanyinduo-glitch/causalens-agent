from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from causalens.tools.causal_tool import run_causal_estimators
from causalens.tools.event_study_tool import (
    check_event_study_readiness,
    run_event_study,
)


def _write_panel(path: Path, *, include_early_lead: bool = True) -> Path:
    rng = np.random.default_rng(20261014)
    periods = [-3, -2, -1, 0, 1, 2]
    if not include_early_lead:
        periods = [-1, 0, 1, 2]
    effects = {0: 1.5, 1: 2.0, 2: 2.5}
    rows: list[dict[str, float | int]] = []
    for unit in range(120):
        treated = int(unit >= 60)
        unit_effect = rng.normal(scale=0.8)
        for period in periods:
            outcome = (
                3.0
                + unit_effect
                + 0.35 * period
                + treated * effects.get(period, 0.0)
                + rng.normal(scale=0.25)
            )
            rows.append(
                {
                    "unit_id": unit,
                    "treated": treated,
                    "event_time": period,
                    "post": int(period >= 0),
                    "outcome": outcome,
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_event_study_recovers_dynamic_panel_effects_and_chart(tmp_path: Path) -> None:
    dataset = _write_panel(tmp_path / "panel.csv")

    response = run_event_study(
        dataset_path=str(dataset),
        group_column="treated",
        event_time_column="event_time",
        outcome="outcome",
        unit_id_column="unit_id",
        post_column="post",
        chart_output_dir=str(tmp_path / "charts"),
    )

    assert response["success"] is True
    result = response["result"]
    effects = {row["event_time"]: row["estimate"] for row in result["dynamic_effects"]}
    assert effects[-1] == 0.0
    assert effects[0] == pytest.approx(1.5, abs=0.2)
    assert effects[1] == pytest.approx(2.0, abs=0.2)
    assert effects[2] == pytest.approx(2.5, abs=0.2)
    assert result["ate"] == pytest.approx(2.0, abs=0.2)
    assert 0.0 <= result["pretrend_test"]["p_value"] <= 1.0
    assert Path(result["chart_path"]).is_file()


def test_event_study_readiness_requires_an_early_lead(tmp_path: Path) -> None:
    dataset = _write_panel(
        tmp_path / "without_lead.csv", include_early_lead=False
    )

    response = check_event_study_readiness(
        dataset_path=str(dataset),
        group_column="treated",
        event_time_column="event_time",
        outcome="outcome",
        unit_id_column="unit_id",
    )

    assert response["success"] is True
    assert response["result"]["ready"] is False
    assert any(
        "早于参考期" in issue
        for issue in response["result"]["blocking_issues"]
    )


def test_suite_selects_event_study_for_multi_period_did(tmp_path: Path) -> None:
    dataset = _write_panel(tmp_path / "suite.csv")

    response = run_causal_estimators(
        dataset_path=str(dataset),
        treatment="treated",
        outcome="outcome",
        confounders=[],
        estimand="ATT",
        design_type="difference_in_differences",
        post_column="post",
        event_time_column="event_time",
        unit_id_column="unit_id",
        chart_output_dir=str(tmp_path / "charts"),
    )

    assert response["success"] is True
    result = response["result"]
    assert result["primary_estimator"] == "EventStudy"
    assert result["estimates"][0]["dynamic_effects"]
    assert result["estimates"][0]["pretrend_test"]["tested_periods"] == [-3, -2]
    grade = result["evidence_grade"]
    assert grade["components"]["design_family"] == "temporal"
    assert "treatment_permutation_diagnostic" not in grade["missing_checks"]
