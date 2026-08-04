from causalens.agent.nodes import (
    _has_causal_analysis_intent,
    _validate_causal_configuration_payload,
    intent_router_node,
)


AVAILABLE_COLUMNS = {
    "user_id",
    "age",
    "income",
    "log_income",
    "historical_activity",
    "channel_score",
    "ad_exposure",
    "retention_score",
}


def test_detects_explicit_causal_effect():
    question = (
        "请估计 ad_exposure 对 "
        "retention_score 的因果效应，"
        "控制 age、log_income、"
        "historical_activity 和 "
        "channel_score。"
    )

    assert (
        _has_causal_analysis_intent(
            question
        )
        is True
    )


def test_conceptual_causal_question_is_not_dml():
    question = (
        "相关关系和因果关系有什么区别？"
    )

    assert (
        _has_causal_analysis_intent(
            question
        )
        is False
    )


def test_router_classifies_causal_analysis():
    result = intent_router_node(
        {
            "question": (
                "请估计 ad_exposure 对 "
                "retention_score 的因果效应，"
                "控制 age、log_income、"
                "historical_activity 和 "
                "channel_score。"
            )
        }
    )

    assert (
        result["task_type"]
        == "causal_analysis"
    )
    assert (
        result["analysis_method"]
        == "linear_dml"
    )


def test_destructive_causal_request_is_unsupported():
    result = intent_router_node(
        {
            "question": (
                "请删除缺失值后执行因果分析。"
            )
        }
    )

    assert (
        result["task_type"]
        == "unsupported"
    )


def test_valid_causal_configuration():
    configuration, errors = (
        _validate_causal_configuration_payload(
            payload={
                "treatment": "ad_exposure",
                "outcome": "retention_score",
                "confounders": [
                    "age",
                    "log_income",
                    "historical_activity",
                    "channel_score",
                ],
            },
            available_columns=(
                AVAILABLE_COLUMNS
            ),
        )
    )

    assert errors == []
    assert configuration is not None
    assert (
        configuration["treatment"]
        == "ad_exposure"
    )


def test_missing_confounders_are_rejected():
    configuration, errors = (
        _validate_causal_configuration_payload(
            payload={
                "treatment": "ad_exposure",
                "outcome": "retention_score",
                "confounders": [],
            },
            available_columns=(
                AVAILABLE_COLUMNS
            ),
        )
    )

    assert configuration is None
    assert any(
        "confounders" in error
        for error in errors
    )


def test_unknown_causal_column_is_rejected():
    configuration, errors = (
        _validate_causal_configuration_payload(
            payload={
                "treatment": "ad_exposure",
                "outcome": "unknown_outcome",
                "confounders": [
                    "age",
                ],
            },
            available_columns=(
                AVAILABLE_COLUMNS
            ),
        )
    )

    assert configuration is None
    assert any(
        "不存在" in error
        for error in errors
    )

def test_ate_inside_column_name_is_not_causal():
    question = (
        "控制 attendance_rate 后，"
        "study_hours 与 score "
        "有什么关系？"
    )

    assert (
        _has_causal_analysis_intent(
            question
        )
        is False
    )

def test_standalone_ate_is_causal():
    question = (
        "请使用这些混杂变量估计 ATE。"
    )

    assert (
        _has_causal_analysis_intent(
            question
        )
        is True
    )