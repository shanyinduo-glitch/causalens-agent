from causalens.agent import nodes


CONFIGURATION = {
    "treatment": "ad_exposure",
    "outcome": "retention_score",
    "confounders": [
        "age",
        "log_income",
        "historical_activity",
        "channel_score",
    ],
}


def test_readiness_node_waits_for_approval(
    monkeypatch,
):
    monkeypatch.setattr(
        nodes,
        "_extract_causal_configuration",
        lambda dataset_path, question: (
            CONFIGURATION,
            None,
        ),
    )

    monkeypatch.setattr(
        nodes,
        "check_causal_readiness",
        lambda **kwargs: {
            "success": True,
            "result": {
                "ready": True,
                "blocking_issues": [],
                "warnings": [
                    "Unobserved confounding "
                    "cannot be ruled out."
                ],
                "sample_summary": {
                    "total_rows": 3000,
                    "complete_cases": 3000,
                    "treatment_counts": {
                        "0": 2141,
                        "1": 859,
                    },
                },
            },
            "error": None,
        },
    )

    result = nodes.causal_readiness_node(
        {
            "dataset_path": (
                "data/synthetic_causal_data.csv"
            ),
            "question": "执行因果分析",
            "step_count": 0,
        }
    )

    assert (
        result["workflow_status"]
        == "awaiting_approval"
    )
    assert result["requires_approval"] is True
    assert result["approved"] is None
    assert (
        result["treatment"]
        == "ad_exposure"
    )
    assert (
        result["causal_readiness"]["ready"]
        is True
    )
    assert (
        result["tool_results"][0]["success"]
        is True
    )


def test_readiness_node_blocks_dml(
    monkeypatch,
):
    monkeypatch.setattr(
        nodes,
        "_extract_causal_configuration",
        lambda dataset_path, question: (
            CONFIGURATION,
            None,
        ),
    )

    monkeypatch.setattr(
        nodes,
        "check_causal_readiness",
        lambda **kwargs: {
            "success": True,
            "result": {
                "ready": False,
                "blocking_issues": [
                    "Sample size is too small."
                ],
                "warnings": [],
                "sample_summary": {
                    "total_rows": 40,
                    "complete_cases": 40,
                    "treatment_counts": {
                        "0": 29,
                        "1": 11,
                    },
                },
            },
            "error": None,
        },
    )

    result = nodes.causal_readiness_node(
        {
            "dataset_path": (
                "data/student_performance.csv"
            ),
            "question": "执行因果分析",
            "step_count": 0,
        }
    )

    assert (
        result["workflow_status"]
        == "blocked"
    )
    assert result["requires_approval"] is False
    assert result["error"] is None
    assert "不会执行 DML" in (
        result["final_answer"]
    )


def test_readiness_node_rejects_missing_config(
    monkeypatch,
):
    monkeypatch.setattr(
        nodes,
        "_extract_causal_configuration",
        lambda dataset_path, question: (
            None,
            "用户没有明确指定 confounders。",
        ),
    )

    result = nodes.causal_readiness_node(
        {
            "dataset_path": (
                "data/synthetic_causal_data.csv"
            ),
            "question": "请做因果分析",
            "step_count": 0,
        }
    )

    assert (
        result["workflow_status"]
        == "failed"
    )
    assert result["requires_approval"] is False
    assert result["error"] is not None