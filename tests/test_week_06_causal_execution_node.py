from causalens.agent import nodes
from causalens.agent.graph import (
    route_after_causal_readiness,
)


READINESS = {
    "ready": True,
    "blocking_issues": [],
    "warnings": [
        "Unobserved confounding "
        "cannot be ruled out."
    ],
}

DIAGNOSTICS = {
    "passed": True,
    "blocking_issues": [],
    "warnings": [],
}


def test_causal_execution_requires_approval():
    result = nodes.causal_execution_node(
        {
            "dataset_path": (
                "data/synthetic_causal_data.csv"
            ),
            "question": "执行因果分析",
            "treatment": "ad_exposure",
            "outcome": "retention_score",
            "confounders": ["age"],
            "causal_readiness": READINESS,
            "approved": None,
        }
    )

    assert (
        result["workflow_status"]
        == "awaiting_approval"
    )
    assert result["error"] is not None


def test_causal_execution_runs_after_approval(
    monkeypatch,
):
    captured_arguments = {}

    def fake_run_causal_estimators(**kwargs):
        captured_arguments.update(kwargs)
        return {
            "success": True,
            "result": {
                "analysis_type": "causal_estimation_suite",
                "primary_estimator": "AIPW",
                "estimates": [
                    {
                        "estimator": "AIPW",
                        "ate": 3.1,
                        "confidence_interval": [2.8, 3.4],
                    },
                ],
            },
            "error": None,
        }

    monkeypatch.setattr(
        nodes,
        "run_causal_estimators",
        fake_run_causal_estimators,
    )

    result = nodes.causal_execution_node(
        {
            "dataset_path": (
                "data/synthetic_causal_data.csv"
            ),
            "question": "执行因果分析",
            "treatment": "ad_exposure",
            "outcome": "retention_score",
            "confounders": [
                "age",
                "log_income",
            ],
            "causal_question": {
                "variable_roles": {
                    "negative_control_outcomes": ["pre_treatment_score"],
                }
            },
            "causal_readiness": READINESS,
            "causal_diagnostics": DIAGNOSTICS,
            "approved": True,
            "step_count": 1,
        }
    )

    assert (
        result["workflow_status"]
        == "reporting"
    )
    assert result["error"] is None
    assert (
        result["tool_results"][0]["success"]
        is True
    )
    assert (
        result["tool_results"][0]
        ["tool_name"]
        == "run_causal_estimators"
    )
    assert captured_arguments["negative_control_outcomes"] == [
        "pre_treatment_score"
    ]


def test_causal_execution_passes_iv_contract_to_suite(monkeypatch) -> None:
    captured = {}

    def fake_run_causal_estimators(**kwargs):
        captured.update(kwargs)
        return {
            "success": True,
            "result": {
                "analysis_type": "causal_estimation_suite",
                "estimand": "LATE",
                "primary_estimator": "TwoStageLeastSquares",
                "estimates": [],
            },
            "error": None,
        }

    monkeypatch.setattr(nodes, "run_causal_estimators", fake_run_causal_estimators)
    result = nodes.causal_execution_node(
        {
            "dataset_path": "data/iv.csv",
            "question": "执行 IV 分析",
            "treatment": "treatment",
            "outcome": "outcome",
            "confounders": ["age"],
            "causal_question": {
                "estimand": "LATE",
                "design_type": "instrumental_variable",
                "variable_roles": {"instruments": ["encouragement"]},
            },
            "causal_readiness": READINESS,
            "causal_diagnostics": DIAGNOSTICS,
            "approved": True,
            "step_count": 1,
        }
    )

    assert result["workflow_status"] == "reporting"
    assert captured["design_type"] == "instrumental_variable"
    assert captured["estimand"] == "LATE"
    assert captured["instruments"] == ["encouragement"]


def test_causal_execution_passes_rdd_contract_to_suite(monkeypatch) -> None:
    captured = {}

    def fake_run_causal_estimators(**kwargs):
        captured.update(kwargs)
        return {
            "success": True,
            "result": {
                "analysis_type": "causal_estimation_suite",
                "estimand": "LATE",
                "primary_estimator": "SharpRegressionDiscontinuity",
                "estimates": [],
            },
            "error": None,
        }

    monkeypatch.setattr(nodes, "run_causal_estimators", fake_run_causal_estimators)
    result = nodes.causal_execution_node(
        {
            "dataset_path": "data/rdd.csv",
            "question": "执行 RDD 分析",
            "treatment": "eligible",
            "outcome": "score",
            "confounders": ["baseline"],
            "causal_question": {
                "estimand": "LATE",
                "design_type": "regression_discontinuity",
                "running_variable": "risk_score",
                "cutoff": 50.0,
                "treatment_above_cutoff": True,
                "bandwidth": 5.0,
                "variable_roles": {},
            },
            "causal_readiness": READINESS,
            "causal_diagnostics": DIAGNOSTICS,
            "approved": True,
            "step_count": 1,
        }
    )

    assert result["workflow_status"] == "reporting"
    assert captured["design_type"] == "regression_discontinuity"
    assert captured["running_variable"] == "risk_score"
    assert captured["cutoff"] == 50.0
    assert captured["bandwidth"] == 5.0


def test_causal_execution_requires_passing_diagnostics(monkeypatch):
    def fail_if_called(**kwargs):
        raise AssertionError("DML must not run without passing diagnostics")

    monkeypatch.setattr(nodes, "run_causal_estimators", fail_if_called)

    result = nodes.causal_execution_node(
        {
            "dataset_path": "data/synthetic_causal_data.csv",
            "question": "执行因果分析",
            "treatment": "ad_exposure",
            "outcome": "retention_score",
            "confounders": ["age"],
            "causal_readiness": READINESS,
            "causal_diagnostics": {
                "passed": False,
                "blocking_issues": ["没有共同支持区间。"],
            },
            "approved": True,
        }
    )

    assert result["workflow_status"] == "blocked"
    assert "研究设计诊断未通过" in result["final_answer"]


def test_causal_execution_passes_cate_contract_to_suite(monkeypatch):
    captured = {}

    def fake_run_causal_estimators(**kwargs):
        captured.update(kwargs)
        return {
            "success": True,
            "result": {
                "analysis_type": "causal_estimation_suite",
                "estimand": "CATE",
                "primary_estimator": "CausalForestDML",
                "estimates": [],
            },
            "error": None,
        }

    monkeypatch.setattr(
        nodes,
        "run_causal_estimators",
        fake_run_causal_estimators,
    )
    result = nodes.causal_execution_node(
        {
            "dataset_path": "data/synthetic_causal_data.csv",
            "question": "估计 age 的异质性效应",
            "treatment": "ad_exposure",
            "outcome": "retention_score",
            "confounders": ["age", "log_income"],
            "causal_question": {
                "estimand": "CATE",
                "variable_roles": {"effect_modifiers": ["age"]},
            },
            "causal_readiness": READINESS,
            "causal_diagnostics": DIAGNOSTICS,
            "approved": True,
        }
    )

    assert result["workflow_status"] == "reporting"
    assert captured["estimand"] == "CATE"
    assert captured["effect_modifiers"] == ["age"]


def test_causal_execution_passes_did_contract_to_suite(monkeypatch):
    captured = {}

    def fake_run_causal_estimators(**kwargs):
        captured.update(kwargs)
        return {
            "success": True,
            "result": {
                "analysis_type": "causal_estimation_suite",
                "estimand": "ATT",
                "primary_estimator": "DifferenceInDifferences",
                "estimates": [],
            },
            "error": None,
        }

    monkeypatch.setattr(
        nodes,
        "run_causal_estimators",
        fake_run_causal_estimators,
    )
    result = nodes.causal_execution_node(
        {
            "dataset_path": "data/did.csv",
            "question": "执行 DID",
            "treatment": "treated_group",
            "outcome": "outcome",
            "confounders": ["baseline"],
            "causal_question": {
                "estimand": "ATT",
                "design_type": "difference_in_differences",
                "post_column": "post",
                "unit_id_column": "unit_id",
                "variable_roles": {"effect_modifiers": []},
            },
            "causal_readiness": READINESS,
            "causal_diagnostics": DIAGNOSTICS,
            "approved": True,
        }
    )

    assert result["workflow_status"] == "reporting"
    assert captured["estimand"] == "ATT"
    assert captured["design_type"] == "difference_in_differences"
    assert captured["post_column"] == "post"
    assert captured["unit_id_column"] == "unit_id"


def test_causal_execution_passes_event_study_contract_to_suite(monkeypatch):
    captured = {}

    def fake_run_causal_estimators(**kwargs):
        captured.update(kwargs)
        return {
            "success": True,
            "result": {
                "analysis_type": "causal_estimation_suite",
                "estimand": "ATT",
                "primary_estimator": "EventStudy",
                "estimates": [],
            },
            "error": None,
        }

    monkeypatch.setattr(nodes, "run_causal_estimators", fake_run_causal_estimators)
    result = nodes.causal_execution_node(
        {
            "dataset_path": "data/event.csv",
            "question": "执行 Event Study",
            "treatment": "treated_group",
            "outcome": "outcome",
            "confounders": [],
            "causal_question": {
                "estimand": "ATT",
                "design_type": "difference_in_differences",
                "event_time_column": "event_time",
                "reference_period": -2,
                "unit_id_column": "unit_id",
                "variable_roles": {"effect_modifiers": []},
            },
            "causal_readiness": READINESS,
            "causal_diagnostics": DIAGNOSTICS,
            "approved": True,
        }
    )

    assert result["workflow_status"] == "reporting"
    assert captured["event_time_column"] == "event_time"
    assert captured["reference_period"] == -2
    assert captured["chart_output_dir"] == "outputs/charts"


def test_causal_execution_passes_staggered_did_contract_to_suite(monkeypatch):
    captured = {}

    def fake_run_causal_estimators(**kwargs):
        captured.update(kwargs)
        return {
            "success": True,
            "result": {
                "analysis_type": "causal_estimation_suite",
                "estimand": "ATT",
                "primary_estimator": "StaggeredAdoptionDID",
                "estimates": [],
            },
            "error": None,
        }

    monkeypatch.setattr(nodes, "run_causal_estimators", fake_run_causal_estimators)
    result = nodes.causal_execution_node(
        {
            "dataset_path": "data/staggered.csv",
            "question": "执行错位处理 DID",
            "treatment": "treated",
            "outcome": "outcome",
            "confounders": [],
            "causal_question": {
                "estimand": "ATT",
                "design_type": "difference_in_differences",
                "time_column": "period",
                "treatment_start_column": "first_treated_period",
                "unit_id_column": "unit_id",
                "variable_roles": {"effect_modifiers": []},
            },
            "causal_readiness": READINESS,
            "causal_diagnostics": DIAGNOSTICS,
            "approved": True,
        }
    )

    assert result["workflow_status"] == "reporting"
    assert captured["time_column"] == "period"
    assert captured["treatment_start_column"] == "first_treated_period"


def test_causal_execution_passes_synthetic_control_contract(monkeypatch):
    captured = {}

    def fake_run_causal_estimators(**kwargs):
        captured.update(kwargs)
        return {
            "success": True,
            "result": {
                "analysis_type": "causal_estimation_suite",
                "estimand": "ATT",
                "primary_estimator": "SyntheticControl",
                "estimates": [],
            },
            "error": None,
        }

    monkeypatch.setattr(nodes, "run_causal_estimators", fake_run_causal_estimators)
    result = nodes.causal_execution_node(
        {
            "dataset_path": "data/panel.csv",
            "question": "执行合成控制",
            "treatment": "treated",
            "outcome": "outcome",
            "confounders": [],
            "causal_question": {
                "estimand": "ATT",
                "design_type": "synthetic_control",
                "time_column": "period",
                "unit_id_column": "unit",
                "intervention_time": 8,
                "treated_unit": "city_a",
                "variable_roles": {"effect_modifiers": []},
            },
            "causal_readiness": READINESS,
            "causal_diagnostics": DIAGNOSTICS,
            "approved": True,
        }
    )

    assert result["workflow_status"] == "reporting"
    assert captured["design_type"] == "synthetic_control"
    assert captured["intervention_time"] == 8
    assert captured["treated_unit"] == "city_a"


def test_route_waits_without_approval():
    destination = (
        route_after_causal_readiness(
            {
                "causal_readiness": READINESS,
                "approved": None,
                "workflow_status": (
                    "awaiting_approval"
                ),
            }
        )
    )

    assert destination == "end"


def test_route_executes_after_approval():
    destination = (
        route_after_causal_readiness(
            {
                "causal_readiness": READINESS,
                "approved": True,
                "workflow_status": (
                    "causal_execution"
                ),
            }
        )
    )

    assert (
        destination
        == "causal_execution"
    )
