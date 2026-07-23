from __future__ import annotations

import pytest

from causalens.agent.graph import (
    route_after_intent,
    run_causalens_graph,
)
from causalens.agent.state import AgentState, TaskType
import json
from itertools import count
from types import SimpleNamespace

DATASET_PATH = "data/student_performance.csv"

def make_fake_chat_completion():
    """
    创建一个确定性的假模型。

    工具决策阶段返回固定、正确的工具参数；
    Reporter 阶段返回固定回答。
    """

    call_ids = count(1)

    def fake_create_chat_completion(
        *,
        messages,
        tools=None,
        tool_choice=None,
        **kwargs,
    ):
        # Reporter 不会传入 tools。
        if not tools:
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content="测试用最终回答。",
                            tool_calls=None,
                        )
                    )
                ]
            )

        tool_name = tools[0]["function"]["name"]

        message_text = "\n".join(
            str(message.get("content", ""))
            for message in messages
        )

        if tool_name == "profile_dataset":
            arguments = {}

        elif tool_name == "retrieve_docs":
            arguments = {
                "query": "passed 字段含义",
                "top_k": 3,
            }

        elif tool_name == "run_readonly_sql":
            arguments = {
                "query": (
                    "SELECT major, "
                    "AVG(score) AS avg_score "
                    "FROM dataset "
                    "GROUP BY major "
                    "ORDER BY major"
                )
            }

        elif tool_name == "create_chart":
            if "分布图" in message_text:
                arguments = {
                    "chart_type": "histogram",
                    "x_column": "score",
                    "title": "Score Distribution",
                }
            else:
                arguments = {
                    "chart_type": "bar",
                    "x_column": "major",
                    "y_column": "score",
                    "aggregation": "mean",
                    "title": "Average Score by Major",
                }

        else:
            raise AssertionError(
                f"测试未处理工具：{tool_name}"
            )

        tool_call = SimpleNamespace(
            id=f"test-call-{next(call_ids)}",
            function=SimpleNamespace(
                name=tool_name,
                arguments=json.dumps(
                    arguments,
                    ensure_ascii=False,
                ),
            ),
        )

        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=None,
                        tool_calls=[tool_call],
                    )
                )
            ]
        )

    return fake_create_chat_completion

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
            "请画出 score 的分布图。",
            "chart_analysis",
        ),
        (
            "请统计不同专业的平均 score，并画柱状图。",
            "mixed_analysis",
        ),
    ],
)
def test_supported_question_runs_full_graph(
    question: str,
    expected_task_type: TaskType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    支持的任务应完成 Router、Planner 和 Reporter 流程。
    """

    monkeypatch.setattr(
        "causalens.agent.nodes.create_chat_completion",
        make_fake_chat_completion(),
    )

    result = run_causalens_graph(
        dataset_path=DATASET_PATH,
        question=question,
    )

    assert result["task_type"] == expected_task_type
    assert result["workflow_status"] == "completed"
    assert result["plan"]
    assert result["final_answer"]
    assert result["error"] is None


def test_unsupported_request_skips_planner() -> None:
    """
    危险请求应直接进入 Reporter，不生成 Planner 计划。
    """

    result = run_causalens_graph(
        dataset_path=DATASET_PATH,
        question="请删除所有 score 小于 60 的学生。",
    )

    assert result["task_type"] == "unsupported"
    assert result["workflow_status"] == "completed"

    # 初始状态中的 plan 是空列表。
    # 因为 Planner 被跳过，所以最终仍应为空。
    assert result["plan"] == []

    assert "不会删除" in result["final_answer"]
    assert "只读数据分析范围" in result["final_answer"]
    assert result["error"] is None


@pytest.mark.parametrize(
    "task_type",
    [
        "data_profile",
        "sql_analysis",
        "chart_analysis",
        "rag_question",
        "mixed_analysis",
    ],
)
def test_route_after_intent_sends_supported_tasks_to_planner(
    task_type: TaskType,
) -> None:
    """
    所有受支持任务都应该进入 Planner。
    """

    state: AgentState = {
        "task_type": task_type,
    }

    assert route_after_intent(state) == "planner"


def test_route_after_intent_sends_unsupported_to_reporter() -> None:
    """
    unsupported 应绕过 Planner，直接进入 Reporter。
    """

    state: AgentState = {
        "task_type": "unsupported",
    }

    assert route_after_intent(state) == "reporter"


def test_initial_state_fields_are_preserved() -> None:
    """
    图执行结束后，应保留输入数据路径、问题和步数上限。
    """

    result = run_causalens_graph(
        dataset_path=DATASET_PATH,
        question="这个数据集有哪些字段？",
        max_steps=7,
    )

    assert result["dataset_path"] == DATASET_PATH
    assert result["question"] == "这个数据集有哪些字段？"
    assert result["max_steps"] == 7
    assert result["step_count"] == 1
    assert len(result["tool_calls"]) == 1
    assert len(result["tool_results"]) == 1
    assert result["tool_results"][0]["success"] is True


def test_graph_executes_sql_tool() -> None:
    """
    SQL 分析任务应该执行 run_readonly_sql，
    并保存调用与结果历史。
    """

    result = run_causalens_graph(
        dataset_path=DATASET_PATH,
        question="不同专业的平均 score 是多少？",
    )

    assert result["workflow_status"] == "completed"
    assert result["pending_tool_call"] is None
    assert result["step_count"] == 1
    assert result["error"] is None

    assert len(result["tool_calls"]) == 1
    assert len(result["tool_results"]) == 1

    tool_call = result["tool_calls"][0]
    tool_result = result["tool_results"][0]

    assert (
        tool_call["tool_name"]
        == "run_readonly_sql"
    )
    assert (
        tool_result["tool_name"]
        == "run_readonly_sql"
    )
    assert tool_result["success"] is True