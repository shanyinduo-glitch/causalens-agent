from __future__ import annotations

import json

import numpy as np
import pandas as pd

from causalens.causal import contract_runner


def _rdd_contract() -> dict:
    return {
        "treatment": "treated",
        "outcome": "outcome",
        "estimand": "LATE",
        "design_type": "regression_discontinuity",
        "running_variable": "score",
        "cutoff": 0,
        "treatment_above_cutoff": True,
        "confounders": [],
        "effect_modifiers": [],
        "mediators": [],
        "colliders": [],
        "instruments": [],
        "negative_control_outcomes": [],
        "target_population": "阈值附近个体",
        "unit_of_analysis": "个体",
        "outcome_horizon": "处理后一期",
        "dag_edges": [],
    }


def _write_rdd_data(path, n: int = 500) -> None:
    rng = np.random.default_rng(2026)
    score = rng.uniform(-1, 1, n)
    treated = (score >= 0).astype(int)
    outcome = 1.5 * treated + 0.7 * score + rng.normal(0, 0.15, n)
    pd.DataFrame(
        {"score": score, "treated": treated, "outcome": outcome}
    ).to_csv(path, index=False)


def test_contract_runner_validates_without_executing_estimator(
    tmp_path, monkeypatch
) -> None:
    dataset = tmp_path / "rdd.csv"
    _write_rdd_data(dataset)

    def fail_if_called(**kwargs):
        raise AssertionError("estimator must not run before approval")

    monkeypatch.setattr(contract_runner, "run_causal_estimators", fail_if_called)
    result = contract_runner.run_explicit_causal_contract(
        dataset_path=dataset,
        contract=_rdd_contract(),
    )

    assert result["success"] is True
    assert result["workflow_status"] == "awaiting_approval"
    assert result["stage"] == "approval"
    assert result["llm_calls"] == 0
    assert result["design_selection"]["selected_estimators"] == [
        "sharp_regression_discontinuity"
    ]


def test_contract_runner_blocks_unknown_columns(tmp_path) -> None:
    dataset = tmp_path / "rdd.csv"
    _write_rdd_data(dataset)
    contract = _rdd_contract()
    contract["running_variable"] = "missing_score"

    result = contract_runner.run_explicit_causal_contract(
        dataset_path=dataset,
        contract=contract,
    )

    assert result["success"] is False
    assert result["workflow_status"] == "blocked"
    assert result["stage"] == "contract_validation"
    assert "missing_score" in " ".join(result["errors"])


def test_contract_runner_executes_real_rdd_file_end_to_end(tmp_path) -> None:
    dataset = tmp_path / "rdd.csv"
    _write_rdd_data(dataset, n=700)

    result = contract_runner.run_explicit_causal_contract(
        dataset_path=dataset,
        contract=_rdd_contract(),
        approved=True,
        result_cache_dir=tmp_path / "cache",
    )

    assert result["success"] is True, json.dumps(result, ensure_ascii=False)
    assert result["workflow_status"] == "completed"
    assert result["estimation"]["success"] is True
    estimates = result["estimation"]["result"]["estimates"]
    assert estimates[0]["estimator"] == "SharpRegressionDiscontinuity"
    assert abs(estimates[0]["ate"] - 1.5) < 0.35


def test_contract_runner_requires_complete_research_context(tmp_path) -> None:
    dataset = tmp_path / "rdd.csv"
    _write_rdd_data(dataset)
    contract = _rdd_contract()
    contract.pop("target_population")

    result = contract_runner.run_explicit_causal_contract(
        dataset_path=dataset,
        contract=contract,
    )

    assert result["workflow_status"] == "blocked"
    assert "target_population" in " ".join(result["errors"])
