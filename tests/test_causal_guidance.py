from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd

from causalens.ui.causal_guidance import (
    assess_guided_compatibility,
    build_guided_contract,
    contract_result_to_agent_state,
    normalize_guided_roles,
    parse_professional_contract,
    profile_columns,
    suggest_causal_roles,
    suggest_causal_roles_with_model,
)


def _financial_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=60).astype(str),
            "y_ret": [0.001 * ((index % 7) - 3) for index in range(60)],
            "t_diff": [0.01 * ((index % 9) - 4) for index in range(60)],
            "comex_gold_ret": [0.002 * ((index % 5) - 2) for index in range(60)],
            "dollar_ret": [0.001 * ((index % 11) - 5) for index in range(60)],
        }
    )


def test_profile_columns_detects_time_and_continuous_fields() -> None:
    profiles = {item["column"]: item for item in profile_columns(_financial_frame())}

    assert profiles["date"]["datetime_like"] is True
    assert profiles["t_diff"]["semantic_type"] == "连续数值"


def test_fuzzy_financial_question_returns_candidates_not_confirmed_roles() -> None:
    suggestions = suggest_causal_roles(
        "分析美国十年期 TIPS 收益率对黄金价格的净因果效应",
        _financial_frame(),
    )

    assert suggestions["treatment"][0] == "t_diff"
    assert "y_ret" in suggestions["outcome"]
    assert "comex_gold_ret" in suggestions["outcome"]
    assert suggestions["time"] == ["date"]


def test_compatibility_blocks_continuous_cross_sectional_treatment() -> None:
    result = assess_guided_compatibility(
        _financial_frame(),
        treatment="t_diff",
        outcome="comex_gold_ret",
        design_type="cross_sectional",
        time_column="date",
    )

    assert result["ready"] is False
    assert result["detected"]["treatment_kind"] == "连续或多水平"
    assert "二元处理变量" in " ".join(result["blocking_issues"])
    assert "时间字段" in " ".join(result["warnings"])


def test_compatibility_accepts_binary_cross_sectional_configuration() -> None:
    frame = pd.DataFrame(
        {
            "treated": [0, 1] * 30,
            "outcome": list(range(60)),
            "age": [20 + index % 10 for index in range(60)],
        }
    )

    result = assess_guided_compatibility(
        frame,
        treatment="treated",
        outcome="outcome",
        design_type="cross_sectional",
    )

    assert result["ready"] is True
    assert result["detected"]["treatment_binary_01"] is True


def test_compatibility_accepts_continuous_time_series_configuration() -> None:
    frame = _financial_frame()
    frame["t_diff"] = [index / 100 for index in range(len(frame))]
    result = assess_guided_compatibility(
        frame,
        treatment="t_diff",
        outcome="comex_gold_ret",
        design_type="time_series",
        time_column="date",
    )

    assert result["ready"] is True
    assert result["detected"]["treatment_kind"] == "连续或多水平"
    assert not result["blocking_issues"]


def test_guided_contract_preserves_explicit_user_choices() -> None:
    contract = build_guided_contract(
        treatment="treated",
        outcome="outcome",
        confounders=["age", "baseline"],
        estimand="ATE",
        design_type="cross_sectional",
        target_population="目标用户",
        unit_of_analysis="用户",
        outcome_horizon="30 天",
    )

    assert contract["treatment"] == "treated"
    assert contract["confounders"] == ["age", "baseline"]
    assert contract["target_population"] == "目标用户"
    assert contract["dag_edges"] == []


def test_guided_contract_excludes_time_column_from_confounders() -> None:
    contract = build_guided_contract(
        treatment="t_diff",
        outcome="y_ret",
        confounders=["date", "dollar_ret"],
        estimand="IRF",
        design_type="time_series",
        target_population="交易日",
        unit_of_analysis="交易日",
        outcome_horizon="未来五日",
        time_column="date",
    )

    assert contract["time_column"] == "date"
    assert contract["confounders"] == ["dollar_ret"]


def test_role_normalizer_reports_all_reserved_design_conflicts() -> None:
    normalized, conflicts = normalize_guided_roles(
        treatment="treated",
        outcome="outcome",
        confounders=["date", "unit", "baseline"],
        effect_modifiers=["outcome", "segment"],
        negative_control_outcomes=["treated", "placebo"],
        time_column="date",
        unit_id_column="unit",
    )

    assert normalized["confounders"] == ["baseline"]
    assert normalized["effect_modifiers"] == ["segment"]
    assert normalized["negative_control_outcomes"] == ["placebo"]
    assert conflicts == {
        "confounders": ["date", "unit"],
        "effect_modifiers": ["outcome"],
        "negative_control_outcomes": ["treated"],
    }


def test_contract_result_adapter_supports_existing_streamlit_renderer() -> None:
    result = {
        "workflow_status": "awaiting_approval",
        "contract": {
            "treatment": "treated",
            "outcome": "outcome",
            "variable_roles": {"confounders": ["age"]},
        },
        "identification": {"status": "not_assessed"},
        "data_quality": {"ready": True},
        "readiness": {"ready": True},
        "diagnostics": {"passed": True},
        "estimation": None,
        "errors": [],
    }

    state = contract_result_to_agent_state(
        result,
        dataset_path="data.csv",
        question="测试问题",
    )

    assert state["task_type"] == "causal_analysis"
    assert state["requires_approval"] is True
    assert state["treatment"] == "treated"
    assert state["tool_results"] == []


def test_professional_contract_parser_returns_friendly_error() -> None:
    parsed, error = parse_professional_contract("[1, 2]")

    assert parsed is None
    assert "顶层必须是对象" in str(error)


def test_model_suggestions_are_filtered_and_remain_unconfirmed(
    monkeypatch,
) -> None:
    import causalens.agent.qwen_client as qwen_client

    payload = {
        "treatment_candidates": ["t_diff", "invented_column"],
        "outcome_candidates": ["comex_gold_ret", "y_ret"],
        "confounder_candidates": ["dollar_ret"],
        "time_candidates": ["date"],
        "clarifying_questions": ["y_ret 和 comex_gold_ret 哪个代表目标？"],
        "warning": "候选需要用户确认。",
    }
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=json.dumps(payload, ensure_ascii=False)
                )
            )
        ]
    )
    monkeypatch.setattr(
        qwen_client,
        "create_chat_completion",
        lambda *args, **kwargs: response,
    )

    result = suggest_causal_roles_with_model(
        "分析 TIPS 收益率对黄金价格的影响",
        _financial_frame(),
    )

    assert result["treatment"] == ["t_diff"]
    assert result["outcome"] == ["comex_gold_ret", "y_ret"]
    assert result["confirmed"] is False
    assert result["llm_calls"] == 1
