from __future__ import annotations

import pytest

from causalens.agent.nodes import (
    intent_router_node,
    planner_node,
    reporter_node,
)
from causalens.agent.state import AgentState, TaskType
from types import SimpleNamespace

@pytest.mark.parametrize(
    ("question", "expected_task_type"),
    [
        (
            "这个数据集有哪些字段？",
            "data_profile",
        ),
        (
            "passed 字段是什么意思？",
            "rag_question",
        ),
        (
            "不同专业的平均 score 是多少？",
            "sql_analysis",
        ),
        (
            "请画出 study_hours 和 score 的散点图。",
            "chart_analysis",
        ),
        (
            "请统计不同专业的平均 score，并画柱状图。",
            "mixed_analysis",
        ),
        (
            "请删除所有 score 小于 60 的学生。",
            "unsupported",
        ),
    ],
)
def test_intent_router_node(
    question: str,
    expected_task_type: TaskType,
) -> None:
    """
    Router 应将常见问题识别为正确的任务类型。
    """

    state: AgentState = {
        "question": question,
        "workflow_status": "routing",
    }

    result = intent_router_node(state)

    assert result["task_type"] == expected_task_type
    assert result["workflow_status"] == "planning"


@pytest.mark.parametrize(
    "task_type",
    [
        "data_profile",
        "sql_analysis",
        "chart_analysis",
        "rag_question",
        "mixed_analysis",
        "unsupported",
    ],
)
def test_planner_node_returns_non_empty_plan(
    task_type: TaskType,
) -> None:
    """
    每一种已定义的任务类型都应该能够生成计划。
    """

    state: AgentState = {
        "question": "测试问题",
        "task_type": task_type,
    }

    result = planner_node(state)

    assert result["workflow_status"] == "tool_decision"
    assert isinstance(result["plan"], list)
    assert len(result["plan"]) > 0
    assert all(
        isinstance(step, str) and step.strip()
        for step in result["plan"]
    )


def test_sql_planner_contains_readonly_sql_step() -> None:
    """
    SQL 分析计划中应体现只读 SQL 查询。
    """

    state: AgentState = {
        "question": "不同专业的平均成绩是多少？",
        "task_type": "sql_analysis",
    }

    result = planner_node(state)

    plan_text = " ".join(result["plan"])

    assert "只读 SQL" in plan_text


def test_rag_planner_contains_document_retrieval_step() -> None:
    """
    RAG 任务计划中应包含知识库或文档检索步骤。
    """

    state: AgentState = {
        "question": "passed 字段是什么意思？",
        "task_type": "rag_question",
    }

    result = planner_node(state)

    plan_text = " ".join(result["plan"])

    assert "知识库" in plan_text
    assert "文档" in plan_text


def test_reporter_node_generates_supported_task_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Reporter 应根据成功的真实工具结果生成最终回答。
    """

    fake_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=(
                        "cs 的平均 score 为 81.23，"
                        "math 为 79.57，"
                        "stats 为 74.69。"
                    ),
                    tool_calls=None,
                )
            )
        ]
    )

    def fake_create_chat_completion(
        **kwargs,
    ):
        return fake_response

    monkeypatch.setattr(
        "causalens.agent.nodes.create_chat_completion",
        fake_create_chat_completion,
    )

    state: AgentState = {
        "question": "不同专业的平均 score 是多少？",
        "task_type": "sql_analysis",
        "plan": [
            "生成只读 SQL 查询",
            "执行查询",
            "总结结果",
        ],
        "tool_results": [
            {
                "tool_name": "run_readonly_sql",
                "success": True,
                "result": {
                    "row_count": 3,
                    "returned_row_count": 3,
                    "truncated": False,
                    "columns": [
                        "major",
                        "avg_score",
                    ],
                    "rows": [
                        {
                            "major": "cs",
                            "avg_score": 81.23,
                        },
                        {
                            "major": "math",
                            "avg_score": 79.57,
                        },
                        {
                            "major": "stats",
                            "avg_score": 74.69,
                        },
                    ],
                },
                "error": None,
            }
        ],
        "error": None,
    }

    result = reporter_node(state)

    assert result["workflow_status"] == "completed"
    assert result.get("error") is None
    assert result["final_answer"]
    assert "81.23" in result["final_answer"]

def test_reporter_node_rejects_unsupported_request() -> None:
    """
    不支持的修改型请求应该得到明确的只读安全说明。
    """

    state: AgentState = {
        "question": "请删除所有 score 小于 60 的学生。",
        "task_type": "unsupported",
        "plan": [],
    }

    result = reporter_node(state)

    assert result["workflow_status"] == "completed"
    assert "不属于 CausaLens 支持的只读数据分析范围" in result["final_answer"]
    assert "不会删除" in result["final_answer"]
    assert "只读 SQL" in result["final_answer"]
    assert "处理计划" not in result["final_answer"]