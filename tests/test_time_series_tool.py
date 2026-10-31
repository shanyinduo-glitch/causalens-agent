from __future__ import annotations

import json

import numpy as np
import pandas as pd

from causalens.causal.contract_runner import run_explicit_causal_contract
from causalens.tools.causal_tool import run_causal_estimators
from causalens.tools.time_series_tool import (
    check_time_series_readiness,
    run_time_series_local_projection,
)


def _write_series(path, *, n: int = 700, duplicate_time: bool = False) -> None:
    rng = np.random.default_rng(20261018)
    shock = rng.normal(0, 1, n)
    control = np.zeros(n)
    outcome = np.zeros(n)
    for index in range(1, n):
        control[index] = 0.4 * control[index - 1] + rng.normal(0, 1)
        outcome[index] = (
            0.45 * outcome[index - 1]
            + 1.8 * shock[index]
            + 0.35 * shock[index - 1]
            + 0.25 * control[index - 1]
            + rng.normal(0, 0.7)
        )
    dates = pd.date_range("2020-01-01", periods=n, freq="D").astype(str).tolist()
    if duplicate_time:
        dates[-1] = dates[-2]
    pd.DataFrame(
        {
            "date": dates,
            "shock": shock,
            "outcome": outcome,
            "control": control,
        }
    ).to_csv(path, index=False)


def _contract() -> dict:
    return {
        "treatment": "shock",
        "outcome": "outcome",
        "estimand": "IRF",
        "design_type": "time_series",
        "time_column": "date",
        "max_horizon": 3,
        "treatment_lags": 1,
        "outcome_lags": 1,
        "control_lag": 1,
        "cumulative_outcome": True,
        "target_population": "该时间区间内的观测期",
        "unit_of_analysis": "单个日期",
        "outcome_horizon": "当期至未来三期累计响应",
        "confounders": ["control"],
        "effect_modifiers": [],
        "mediators": [],
        "colliders": [],
        "instruments": [],
        "negative_control_outcomes": [],
        "dag_edges": [],
    }


def test_time_series_readiness_accepts_ordered_continuous_shock(tmp_path) -> None:
    path = tmp_path / "series.csv"
    _write_series(path)

    response = check_time_series_readiness(
        dataset_path=str(path),
        treatment="shock",
        outcome="outcome",
        time_column="date",
        confounders=["control"],
        max_horizon=3,
    )

    assert response["success"] is True
    assert response["result"]["ready"] is True
    assert response["result"]["sample_summary"]["complete_rows"] == 700


def test_time_series_readiness_blocks_duplicate_time_points(tmp_path) -> None:
    path = tmp_path / "series.csv"
    _write_series(path, duplicate_time=True)

    response = check_time_series_readiness(
        dataset_path=str(path),
        treatment="shock",
        outcome="outcome",
        time_column="date",
        confounders=["control"],
    )

    assert response["success"] is True
    assert response["result"]["ready"] is False
    assert "duplicate" in " ".join(response["result"]["blocking_issues"])


def test_local_projection_recovers_impact_response(tmp_path) -> None:
    path = tmp_path / "series.csv"
    _write_series(path)

    response = run_time_series_local_projection(
        dataset_path=str(path),
        treatment="shock",
        outcome="outcome",
        time_column="date",
        confounders=["control"],
        max_horizon=3,
    )

    assert response["success"] is True, response
    result = response["result"]
    assert abs(result["ate"] - 1.8) < 0.20
    assert len(result["dynamic_effects"]) == 4
    assert result["dynamic_effects"][1]["estimate"] > result["ate"]
    assert result["model_configuration"]["covariance"] == "Newey-West HAC"


def test_time_series_design_runs_through_estimator_suite(tmp_path) -> None:
    path = tmp_path / "series.csv"
    _write_series(path)

    response = run_causal_estimators(
        dataset_path=str(path),
        treatment="shock",
        outcome="outcome",
        confounders=["control"],
        estimand="IRF",
        design_type="time_series",
        time_column="date",
        max_horizon=2,
    )

    assert response["success"] is True, response
    result = response["result"]
    assert result["primary_estimator"] == "TimeSeriesLocalProjection"
    assert result["design_selection"]["data_structure"] == "time_series"
    assert result["estimates"][0]["sensitivity_analysis"]["available"] is False


def test_explicit_contract_runs_time_series_without_llm(tmp_path) -> None:
    path = tmp_path / "series.csv"
    _write_series(path)

    validation = run_explicit_causal_contract(
        dataset_path=path,
        contract=_contract(),
        approved=False,
    )
    completed = run_explicit_causal_contract(
        dataset_path=path,
        contract=_contract(),
        approved=True,
    )

    assert validation["workflow_status"] == "awaiting_approval", json.dumps(
        validation, ensure_ascii=False
    )
    assert validation["llm_calls"] == 0
    assert completed["workflow_status"] == "completed", json.dumps(
        completed, ensure_ascii=False
    )
    assert completed["estimation"]["result"]["estimand"] == "IRF"
