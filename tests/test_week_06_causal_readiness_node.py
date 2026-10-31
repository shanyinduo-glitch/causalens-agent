import pandas as pd

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

IV_CONFIGURATION = {
    "treatment": "treatment",
    "outcome": "outcome",
    "confounders": ["age"],
    "causal_question": {
        "treatment": "treatment",
        "outcome": "outcome",
        "estimand": "LATE",
        "design_type": "instrumental_variable",
        "target_population": "users",
        "unit_of_analysis": "user",
        "outcome_horizon": "30 days",
        "variable_roles": {
            "confounders": ["age"],
            "instruments": ["encouragement"],
        },
    },
}

RDD_CONFIGURATION = {
    "treatment": "eligible",
    "outcome": "score",
    "confounders": ["baseline"],
    "causal_question": {
        "treatment": "eligible",
        "outcome": "score",
        "estimand": "LATE",
        "design_type": "regression_discontinuity",
        "running_variable": "risk_score",
        "cutoff": 50.0,
        "treatment_above_cutoff": True,
        "target_population": "applicants",
        "unit_of_analysis": "applicant",
        "outcome_horizon": "one year",
        "variable_roles": {"confounders": ["baseline"]},
    },
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
    assert result["causal_diagnostics"]["passed"] is True
    assert "propensity_summary" in result["causal_diagnostics"]


def test_iv_readiness_node_uses_first_stage_instead_of_propensity(
    monkeypatch,
) -> None:
    captured = {}
    monkeypatch.setattr(
        nodes,
        "_extract_causal_configuration",
        lambda dataset_path, question: (IV_CONFIGURATION, None),
    )

    def fake_check_iv_readiness(**kwargs):
        captured.update(kwargs)
        return {
            "success": True,
            "result": {
                "ready": True,
                "blocking_issues": [],
                "warnings": [],
                "sample_summary": {
                    "total_rows": 500,
                    "complete_cases": 500,
                },
                "first_stage": {
                    "f_statistic": 42.0,
                    "partial_r2": 0.22,
                    "weak_instrument_warning": False,
                },
            },
            "error": None,
        }

    monkeypatch.setattr(nodes, "check_iv_readiness", fake_check_iv_readiness)
    result = nodes.causal_readiness_node(
        {
            "dataset_path": "data/iv.csv",
            "question": "执行 IV 分析",
            "step_count": 0,
        }
    )

    assert result["workflow_status"] == "awaiting_approval"
    assert result["tool_calls"][0]["tool_name"] == "check_iv_readiness"
    assert result["causal_diagnostics"]["analysis_type"] == (
        "iv_design_diagnostics"
    )
    assert captured["instruments"] == ["encouragement"]
    assert "first-stage F" in result["final_answer"]


def test_rdd_readiness_node_uses_cutoff_diagnostics(monkeypatch) -> None:
    captured = {}
    monkeypatch.setattr(
        nodes,
        "_extract_causal_configuration",
        lambda dataset_path, question: (RDD_CONFIGURATION, None),
    )

    def fake_check_rdd_readiness(**kwargs):
        captured.update(kwargs)
        return {
            "success": True,
            "result": {
                "ready": True,
                "bandwidth": 5.0,
                "blocking_issues": [],
                "warnings": [],
                "sample_summary": {
                    "total_rows": 600,
                    "complete_cases": 600,
                    "assignment_compliance": 1.0,
                },
                "density_diagnostic": {"p_value": 0.4, "passed": True},
                "covariate_continuity": {"all_passed": True},
            },
            "error": None,
        }

    monkeypatch.setattr(nodes, "check_rdd_readiness", fake_check_rdd_readiness)
    result = nodes.causal_readiness_node(
        {
            "dataset_path": "data/rdd.csv",
            "question": "执行 RDD 分析",
            "step_count": 0,
        }
    )

    assert result["workflow_status"] == "awaiting_approval"
    assert result["tool_calls"][0]["tool_name"] == "check_rdd_readiness"
    assert result["causal_diagnostics"]["analysis_type"] == (
        "rdd_design_diagnostics"
    )
    assert captured["running_variable"] == "risk_score"
    assert captured["cutoff"] == 50.0
    assert "assignment compliance" in result["final_answer"]


def test_readiness_node_blocks_when_overlap_fails(
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
                "warnings": [],
                "sample_summary": {
                    "total_rows": 3000,
                    "complete_cases": 3000,
                    "treatment_counts": {"0": 2141, "1": 859},
                },
            },
            "error": None,
        },
    )
    monkeypatch.setattr(
        nodes,
        "run_causal_diagnostics",
        lambda **kwargs: {
            "success": True,
            "result": {
                "passed": False,
                "blocking_issues": [
                    "处理组与对照组的倾向得分没有共同支持区间。"
                ],
                "warnings": [],
            },
            "error": None,
        },
    )

    result = nodes.causal_readiness_node(
        {
            "dataset_path": "data/synthetic_causal_data.csv",
            "question": "执行因果分析",
            "step_count": 0,
        }
    )

    assert result["workflow_status"] == "blocked"
    assert result["requires_approval"] is False
    assert result["causal_diagnostics"]["passed"] is False
    assert "共同支持区间" in result["final_answer"]


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
    assert "不会执行因果效应估计" in (
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


def test_readiness_node_blocks_unsupported_estimand(monkeypatch):
    configuration = {
        **CONFIGURATION,
        "causal_question": {
            "treatment": "ad_exposure",
            "outcome": "retention_score",
            "estimand": "ATT",
            "treatment_contrast": {"control": 0, "treated": 1},
            "target_population": "当前用户",
            "unit_of_analysis": "user",
            "outcome_horizon": "曝光后 30 天",
            "variable_roles": {
                "confounders": CONFIGURATION["confounders"],
            },
        },
    }
    monkeypatch.setattr(
        nodes,
        "_extract_causal_configuration",
        lambda dataset_path, question: (configuration, None),
    )

    def fail_if_called(**kwargs):
        raise AssertionError("readiness must not run for unsupported estimand")

    monkeypatch.setattr(nodes, "check_causal_readiness", fail_if_called)

    result = nodes.causal_readiness_node(
        {
            "dataset_path": "data/synthetic_causal_data.csv",
            "question": "估计 ATT",
            "step_count": 0,
        }
    )

    assert result["workflow_status"] == "blocked"
    assert result["requires_approval"] is False
    assert "ATT" in result["final_answer"]
    assert "ATE" in result["final_answer"]


def test_readiness_node_blocks_cate_without_effect_modifier(monkeypatch):
    configuration = {
        **CONFIGURATION,
        "causal_question": {
            "treatment": "ad_exposure",
            "outcome": "retention_score",
            "estimand": "CATE",
            "treatment_contrast": {"control": 0, "treated": 1},
            "target_population": "当前用户",
            "unit_of_analysis": "user",
            "outcome_horizon": "曝光后 30 天",
            "variable_roles": {
                "confounders": CONFIGURATION["confounders"],
                "effect_modifiers": [],
            },
        },
    }
    monkeypatch.setattr(
        nodes,
        "_extract_causal_configuration",
        lambda dataset_path, question: (configuration, None),
    )

    result = nodes.causal_readiness_node(
        {
            "dataset_path": "data/synthetic_causal_data.csv",
            "question": "估计 CATE",
            "step_count": 0,
        }
    )

    assert result["workflow_status"] == "blocked"
    assert result["requires_approval"] is False
    assert "effect modifier" in result["final_answer"]


def test_readiness_node_accepts_cate_with_effect_modifier(monkeypatch):
    configuration = {
        **CONFIGURATION,
        "causal_question": {
            "treatment": "ad_exposure",
            "outcome": "retention_score",
            "estimand": "CATE",
            "treatment_contrast": {"control": 0, "treated": 1},
            "target_population": "当前用户",
            "unit_of_analysis": "user",
            "outcome_horizon": "曝光后 30 天",
            "variable_roles": {
                "confounders": CONFIGURATION["confounders"],
                "effect_modifiers": ["age"],
            },
        },
    }
    monkeypatch.setattr(
        nodes,
        "_extract_causal_configuration",
        lambda dataset_path, question: (configuration, None),
    )

    result = nodes.causal_readiness_node(
        {
            "dataset_path": "data/synthetic_causal_data.csv",
            "question": "估计 age 修饰的 CATE",
            "step_count": 0,
        }
    )

    assert result["workflow_status"] == "awaiting_approval"
    assert result["requires_approval"] is True
    assert "CausalForestDML" in result["final_answer"]
    assert "effect modifiers：['age']" in result["final_answer"]


def test_readiness_node_accepts_explicit_did_design(monkeypatch, tmp_path):
    rows = []
    for group in (0, 1):
        for post in (0, 1):
            for index in range(30):
                rows.append(
                    {
                        "treated_group": group,
                        "post": post,
                        "outcome": 5 + post + 2 * group * post + index / 100,
                        "baseline": index,
                    }
                )
    dataset_path = tmp_path / "did.csv"
    pd.DataFrame(rows).to_csv(dataset_path, index=False)
    configuration = {
        "treatment": "treated_group",
        "outcome": "outcome",
        "confounders": ["baseline"],
        "causal_question": {
            "treatment": "treated_group",
            "outcome": "outcome",
            "estimand": "ATT",
            "design_type": "difference_in_differences",
            "post_column": "post",
            "unit_id_column": None,
            "treatment_contrast": {"control": 0, "treated": 1},
            "target_population": "样本总体",
            "unit_of_analysis": "observation",
            "outcome_horizon": "post period",
            "variable_roles": {"confounders": ["baseline"]},
        },
    }
    monkeypatch.setattr(
        nodes,
        "_extract_causal_configuration",
        lambda dataset_path, question: (configuration, None),
    )

    result = nodes.causal_readiness_node(
        {
            "dataset_path": str(dataset_path),
            "question": "执行 DID",
            "step_count": 0,
        }
    )

    assert result["workflow_status"] == "awaiting_approval"
    assert result["requires_approval"] is True
    assert result["tool_calls"][0]["tool_name"] == "check_did_readiness"
    assert result["causal_diagnostics"]["analysis_type"] == (
        "did_design_diagnostics"
    )
    assert "Difference-in-Differences" in result["final_answer"]


def test_readiness_node_routes_multi_period_did_to_event_study(
    monkeypatch, tmp_path
):
    rows = []
    for unit in range(40):
        group = int(unit >= 20)
        for period in (-3, -2, -1, 0, 1, 2):
            rows.append(
                {
                    "unit_id": unit,
                    "treated_group": group,
                    "event_time": period,
                    "outcome": 4 + period + group * max(period + 1, 0),
                }
            )
    dataset_path = tmp_path / "event.csv"
    pd.DataFrame(rows).to_csv(dataset_path, index=False)
    configuration = {
        "treatment": "treated_group",
        "outcome": "outcome",
        "confounders": [],
        "causal_question": {
            "treatment": "treated_group",
            "outcome": "outcome",
            "estimand": "ATT",
            "design_type": "difference_in_differences",
            "event_time_column": "event_time",
            "reference_period": -1,
            "unit_id_column": "unit_id",
            "treatment_contrast": {"control": 0, "treated": 1},
            "target_population": "样本总体",
            "unit_of_analysis": "unit",
            "outcome_horizon": "event time -3 to 2",
            "variable_roles": {"confounders": []},
        },
    }
    monkeypatch.setattr(
        nodes,
        "_extract_causal_configuration",
        lambda dataset_path, question: (configuration, None),
    )

    result = nodes.causal_readiness_node(
        {
            "dataset_path": str(dataset_path),
            "question": "执行 Event Study",
            "step_count": 0,
        }
    )

    assert result["workflow_status"] == "awaiting_approval"
    assert result["tool_calls"][0]["tool_name"] == "check_event_study_readiness"
    assert result["causal_diagnostics"]["analysis_type"] == (
        "event_study_design_diagnostics"
    )
    assert "Event Study" in result["final_answer"]


def test_readiness_node_blocks_invalid_adjustment_set(monkeypatch):
    configuration = {
        "treatment": "ad_exposure",
        "outcome": "retention_score",
        "confounders": ["channel_score"],
        "causal_dag_edges": [
            ("age", "ad_exposure"),
            ("age", "retention_score"),
            ("channel_score", "retention_score"),
            ("ad_exposure", "retention_score"),
        ],
    }
    monkeypatch.setattr(
        nodes,
        "_extract_causal_configuration",
        lambda dataset_path, question: (configuration, None),
    )

    def fail_if_called(**kwargs):
        raise AssertionError("technical readiness must not run")

    monkeypatch.setattr(nodes, "check_causal_readiness", fail_if_called)
    result = nodes.causal_readiness_node(
        {
            "dataset_path": "data/synthetic_causal_data.csv",
            "question": "执行因果分析",
            "step_count": 0,
        }
    )

    assert result["workflow_status"] == "blocked"
    assert result["causal_identification"]["identified"] is False
    assert "后门路径" in result["final_answer"]


def test_readiness_node_routes_synthetic_control(monkeypatch, tmp_path):
    rows = []
    for unit in range(6):
        for period in range(10):
            rows.append(
                {
                    "unit": f"u{unit}",
                    "period": period,
                    "treated": int(unit == 0 and period >= 6),
                    "outcome": 3 + 0.2 * period + 0.1 * unit
                    + (2 if unit == 0 and period >= 6 else 0),
                }
            )
    dataset_path = tmp_path / "synthetic_control.csv"
    pd.DataFrame(rows).to_csv(dataset_path, index=False)
    configuration = {
        "treatment": "treated",
        "outcome": "outcome",
        "confounders": [],
        "causal_question": {
            "treatment": "treated",
            "outcome": "outcome",
            "estimand": "ATT",
            "design_type": "synthetic_control",
            "unit_id_column": "unit",
            "time_column": "period",
            "intervention_time": 6,
            "treatment_contrast": {"control": 0, "treated": 1},
            "target_population": "cities",
            "unit_of_analysis": "city",
            "outcome_horizon": "four periods",
            "variable_roles": {"confounders": []},
        },
    }
    monkeypatch.setattr(
        nodes,
        "_extract_causal_configuration",
        lambda dataset_path, question: (configuration, None),
    )

    result = nodes.causal_readiness_node(
        {
            "dataset_path": str(dataset_path),
            "question": "执行合成控制",
            "step_count": 0,
        }
    )

    assert result["workflow_status"] == "awaiting_approval"
    assert result["tool_calls"][0]["tool_name"] == (
        "check_synthetic_control_readiness"
    )
    assert result["causal_data_quality"]["ready"] is True
    assert "SyntheticControl" in result["final_answer"]


def test_readiness_node_routes_continuous_time_series_irf(monkeypatch, tmp_path):
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=120, freq="D"),
            "shock": [((index * 17) % 43 - 21) / 100 for index in range(120)],
            "outcome": [((index * 11) % 37 - 18) / 100 for index in range(120)],
            "control": [((index * 7) % 31 - 15) / 100 for index in range(120)],
        }
    )
    dataset_path = tmp_path / "time_series.csv"
    frame.to_csv(dataset_path, index=False)
    configuration = {
        "treatment": "shock",
        "outcome": "outcome",
        "confounders": ["control"],
        "causal_question": {
            "treatment": "shock",
            "outcome": "outcome",
            "estimand": "IRF",
            "design_type": "time_series",
            "time_column": "date",
            "max_horizon": 4,
            "treatment_lags": 1,
            "outcome_lags": 1,
            "control_lag": 1,
            "cumulative_outcome": True,
            "treatment_contrast": {"control": 0, "treated": 1},
            "target_population": "sample period",
            "unit_of_analysis": "day",
            "outcome_horizon": "four days",
            "variable_roles": {"confounders": ["control"]},
        },
    }
    monkeypatch.setattr(
        nodes,
        "_extract_causal_configuration",
        lambda dataset_path, question: (configuration, None),
    )

    result = nodes.causal_readiness_node(
        {
            "dataset_path": str(dataset_path),
            "question": "估计连续冲击的时间序列动态响应",
            "step_count": 0,
        }
    )

    assert result["workflow_status"] == "awaiting_approval"
    assert result["tool_calls"][0]["tool_name"] == "check_time_series_readiness"
    assert result["causal_data_quality"]["ready"] is True
    assert "TimeSeriesLocalProjection" in result["final_answer"]
    assert "不是二元处理 ATE" in result["final_answer"]
