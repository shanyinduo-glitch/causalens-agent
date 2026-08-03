from causalens.agent.nodes import (
    _infer_required_tool_names,
    _infer_statistical_tool_name,
    intent_router_node,
    planner_node,
)

def test_infers_correlation_tool():
    question = (
        "study_hours 和 score 是否相关？"
    )

    assert (
        _infer_statistical_tool_name(
            question
        )
        == "run_correlation"
    )

    assert _infer_required_tool_names(
        task_type="statistical_analysis",
        question=question,
    ) == [
        "run_correlation"
    ]


def test_infers_group_comparison_tool():
    question = (
        "不同 major 的 score "
        "是否存在显著差异？"
    )

    assert (
        _infer_statistical_tool_name(
            question
        )
        == "run_group_comparison"
    )

    assert _infer_required_tool_names(
        task_type="statistical_analysis",
        question=question,
    ) == [
        "run_group_comparison"
    ]


def test_infers_ols_tool():
    question = (
        "控制 attendance_rate 后，"
        "study_hours 与 score "
        "有什么关系？"
    )

    assert (
        _infer_statistical_tool_name(
            question
        )
        == "run_ols"
    )

    assert _infer_required_tool_names(
        task_type="statistical_analysis",
        question=question,
    ) == [
        "run_ols"
    ]


def test_ols_has_priority_over_correlation():
    question = (
        "控制 attendance_rate 后，"
        "分析 study_hours 与 score "
        "的相关关系。"
    )

    assert (
        _infer_statistical_tool_name(
            question
        )
        == "run_ols"
    )


def test_unknown_statistical_method_returns_none():
    assert (
        _infer_statistical_tool_name(
            "请进行统计分析。"
        )
        is None
    )


def test_causal_task_does_not_directly_select_dml():
    required_tools = (
        _infer_required_tool_names(
            task_type="causal_analysis",
            question=(
                "广告曝光是否导致留存提高？"
            ),
        )
    )

    assert required_tools == []

def test_router_classifies_correlation():
    result = intent_router_node(
        {
            "question": (
                "study_hours 和 score "
                "是否相关？"
            ),
        }
    )

    assert (
        result["task_type"]
        == "statistical_analysis"
    )
    assert (
        result["analysis_method"]
        == "run_correlation"
    )


def test_router_classifies_group_comparison():
    result = intent_router_node(
        {
            "question": (
                "不同 major 的平均 score "
                "是否存在显著差异？"
            ),
        }
    )

    assert (
        result["task_type"]
        == "statistical_analysis"
    )
    assert (
        result["analysis_method"]
        == "run_group_comparison"
    )


def test_router_classifies_ols():
    result = intent_router_node(
        {
            "question": (
                "控制 attendance_rate 后，"
                "study_hours 与 score "
                "有什么关系？"
            ),
        }
    )

    assert (
        result["task_type"]
        == "statistical_analysis"
    )
    assert (
        result["analysis_method"]
        == "run_ols"
    )


def test_router_keeps_plain_average_as_sql():
    result = intent_router_node(
        {
            "question": (
                "不同专业的平均 score "
                "分别是多少？"
            ),
        }
    )

    assert (
        result["task_type"]
        == "sql_analysis"
    )


def test_router_keeps_destructive_request_unsupported():
    result = intent_router_node(
        {
            "question": (
                "请删除数据后再进行回归分析。"
            ),
        }
    )

    assert (
        result["task_type"]
        == "unsupported"
    )

def test_planner_builds_correlation_plan():
    result = planner_node(
        {
            "task_type": "statistical_analysis",
            "analysis_method": "run_correlation",
        }
    )

    plan = result["plan"]

    assert any(
        "Pearson" in step
        or "Spearman" in step
        for step in plan
    )
    assert any(
        "因果" in step
        for step in plan
    )
    assert (
        result["workflow_status"]
        == "tool_decision"
    )


def test_planner_builds_group_comparison_plan():
    result = planner_node(
        {
            "task_type": "statistical_analysis",
            "analysis_method": (
                "run_group_comparison"
            ),
        }
    )

    plan = result["plan"]

    assert any(
        "t 检验" in step
        or "方差分析" in step
        for step in plan
    )


def test_planner_builds_ols_plan():
    result = planner_node(
        {
            "task_type": "statistical_analysis",
            "analysis_method": "run_ols",
        }
    )

    plan = result["plan"]

    assert any(
        "稳健标准误" in step
        for step in plan
    )
    assert any(
        "条件关联" in step
        for step in plan
    )


def test_planner_handles_unknown_statistical_method():
    result = planner_node(
        {
            "task_type": "statistical_analysis",
            "analysis_method": None,
        }
    )

    assert result["plan"]
    assert (
        result["workflow_status"]
        == "tool_decision"
    )


def test_planner_has_causal_safety_plan():
    result = planner_node(
        {
            "task_type": "causal_analysis",
            "analysis_method": None,
        }
    )

    plan = result["plan"]

    assert any(
        "准备度" in step
        for step in plan
    )
    assert any(
        "确认" in step
        for step in plan
    )
