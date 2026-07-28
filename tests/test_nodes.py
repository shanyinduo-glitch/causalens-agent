from __future__ import annotations

import pytest

from causalens.agent.nodes import (
    intent_router_node,
    planner_node,
    reporter_node,
    tool_decision_node,
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
        (
                "study_hours 的单位是什么？",
                "rag_question",
        ),
        (
                "请画出 study_hours 和 score 的散点图。",
                "chart_analysis",
        ),
        (
                "请说明 study_hours 的单位，"
                "并画出 study_hours 和 score 的散点图。",
                "mixed_analysis",
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


def test_tool_decision_extracts_rag_subquestion_for_mixed_analysis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    混合任务执行 retrieve_docs 时，
    应只保留字段定义部分，避免统计需求干扰文档检索。
    """

    fake_tool_call = SimpleNamespace(
        id="call_test_mixed_rag",
        function=SimpleNamespace(
            name="retrieve_docs",
            arguments=(
                '{"query": "passed 字段含义 数据字典", '
                '"top_k": 3}'
            ),
        ),
    )

    fake_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=None,
                    tool_calls=[fake_tool_call],
                )
            )
        ]
    )

    def fake_create_chat_completion(
        messages,
        *,
        tools=None,
        tool_choice=None,
    ):
        return fake_response

    monkeypatch.setattr(
        "causalens.agent.nodes.create_chat_completion",
        fake_create_chat_completion,
    )

    state: AgentState = {
        "dataset_path": "data/student_performance.csv",
        "question": (
            "请解释 passed 字段含义，"
            "并统计不同 passed 值的平均 score。"
        ),
        "task_type": "mixed_analysis",
        "plan": [
            "检索 passed 字段定义",
            "统计不同 passed 值的平均 score",
        ],
        "required_tools": [
            "retrieve_docs",
            "run_readonly_sql",
        ],
        "tool_results": [],
        "validation_error": "",
    }

    result = tool_decision_node(state)

    assert result.get("error") is None
    assert result["workflow_status"] == "tool_validation"
    assert result["required_tools"] == [
        "retrieve_docs",
        "run_readonly_sql",
    ]

    pending_tool_call = result["pending_tool_call"]

    assert pending_tool_call is not None
    assert pending_tool_call["name"] == "retrieve_docs"
    assert pending_tool_call["arguments"]["query"] == (
        "请解释 passed 字段含义"
    )
    assert pending_tool_call["arguments"]["top_k"] == 3

def test_tool_decision_preserves_original_query_for_rag_question(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    单一 RAG 问题应直接使用用户原始问题作为检索 query。

    即使模型擅自添加数据集名称或堆叠关键词，
    Python 规则也应覆盖模型生成的 query，
    避免降低向量检索准确性。
    """

    fake_tool_call = SimpleNamespace(
        id="call_test_rag",
        function=SimpleNamespace(
            name="retrieve_docs",
            arguments=(
                '{"query": '
                '"student_performance.csv passed 字段 定义 含义", '
                '"top_k": 3}'
            ),
        ),
    )

    fake_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=None,
                    tool_calls=[fake_tool_call],
                )
            )
        ]
    )


    def fake_create_chat_completion(
        messages,
        *,
        tools=None,
        tool_choice=None,
    ):
        return fake_response

    monkeypatch.setattr(
        "causalens.agent.nodes.create_chat_completion",
        fake_create_chat_completion,
    )

    state: AgentState = {
        "dataset_path": "data/student_performance.csv",
        "question": "passed 字段是什么意思？",
        "task_type": "rag_question",
        "plan": [
            "从项目知识库检索字段定义",
            "基于文档证据回答问题",
        ],
        "required_tools": ["retrieve_docs"],
        "tool_results": [],
        "validation_error": "",
    }

    result = tool_decision_node(state)

    assert result.get("error") is None
    assert result["workflow_status"] == "tool_validation"
    assert result["required_tools"] == ["retrieve_docs"]

    pending_tool_call = result["pending_tool_call"]

    assert pending_tool_call is not None
    assert pending_tool_call["name"] == "retrieve_docs"
    assert pending_tool_call["arguments"]["query"] == (
        "passed 字段是什么意思？"
    )
    assert pending_tool_call["arguments"]["top_k"] == 3


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


def test_reporter_forbids_unsupported_histogram_shape_inference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    当直方图只返回基础统计量时，
    Reporter 提示词应明确禁止推断偏态、对称性和集中程度。
    """

    captured_messages = []

    fake_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=(
                        "score 共 40 条记录，均值为 78.525，"
                        "中位数为 79.5，范围为 52 到 98。"
                    ),
                    tool_calls=None,
                )
            )
        ]
    )

    def fake_create_chat_completion(
        messages,
        *,
        tools=None,
        tool_choice=None,
    ):
        captured_messages.extend(messages)
        return fake_response

    monkeypatch.setattr(
        "causalens.agent.nodes.create_chat_completion",
        fake_create_chat_completion,
    )

    state: AgentState = {
        "question": "请画出 score 的分布图。",
        "task_type": "chart_analysis",
        "plan": [
            "识别图表类型",
            "生成图表",
            "总结结果",
        ],
        "tool_results": [
            {
                "tool_name": "create_chart",
                "success": True,
                "result": {
                    "chart_type": "histogram",
                    "chart_path": "outputs/charts/score.png",
                    "plotted_row_count": 40,
                    "numeric_summary": {
                        "count": 40,
                        "mean": 78.525,
                        "median": 79.5,
                        "min": 52.0,
                        "max": 98.0,
                    },
                },
                "error": None,
            }
        ],
        "error": None,
    }

    result = reporter_node(state)

    assert result["workflow_status"] == "completed"
    assert result.get("error") is None

    system_prompt = captured_messages[0]["content"]

    assert "不得判断偏态" in system_prompt
    assert "对称性" in system_prompt
    assert "集中程度" in system_prompt
    assert "峰值区间" in system_prompt
    assert "长尾特征" in system_prompt


def test_reporter_requires_scatter_relationship_summary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    散点图返回 relationship_summary 时，
    Reporter 提示词应要求总结相关系数并保留因果边界。
    """

    captured_messages = []

    fake_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=(
                        "共绘制 40 个数据点，Pearson 相关系数为 0.9874，"
                        "表明当前样本中存在很强的正向线性关联。"
                        "相关性不能证明因果关系。"
                    ),
                    tool_calls=None,
                )
            )
        ]
    )

    def fake_create_chat_completion(
        messages,
        *,
        tools=None,
        tool_choice=None,
    ):
        captured_messages.extend(messages)
        return fake_response

    monkeypatch.setattr(
        "causalens.agent.nodes.create_chat_completion",
        fake_create_chat_completion,
    )

    state: AgentState = {
        "question": (
            "请说明 study_hours 的单位，"
            "并画出 study_hours 和 score 的散点图。"
        ),
        "task_type": "mixed_analysis",
        "plan": [
            "检索字段单位",
            "生成散点图",
            "汇总结果",
        ],
        "tool_results": [
            {
                "tool_name": "retrieve_docs",
                "success": True,
                "result": {
                    "query": "请说明 study_hours 的单位",
                    "results": [],
                },
                "error": None,
            },
            {
                "tool_name": "create_chart",
                "success": True,
                "result": {
                    "chart_type": "scatter",
                    "chart_path": "outputs/charts/scatter.png",
                    "relationship_summary": {
                        "point_count": 40,
                        "x_min": 3.0,
                        "x_max": 18.0,
                        "y_min": 52.0,
                        "y_max": 98.0,
                        "pearson_correlation": 0.9874,
                    },
                },
                "error": None,
            },
        ],
        "error": None,
    }

    result = reporter_node(state)

    assert result["workflow_status"] == "completed"

    system_prompt = captured_messages[0]["content"]

    assert "relationship_summary" in system_prompt
    assert "Pearson 相关系数" in system_prompt
    assert "相关性不能证明因果关系" in system_prompt