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
    monkeypatch.setattr(
        nodes,
        "run_dml",
        lambda **kwargs: {
            "success": True,
            "result": {
                "analysis_type": (
                    "linear_dml"
                ),
                "ate": 3.14,
                "confidence_interval": [
                    2.80,
                    3.48,
                ],
            },
            "error": None,
        },
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
            "causal_readiness": READINESS,
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
        == "run_dml"
    )


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