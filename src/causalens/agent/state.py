from __future__ import annotations

from operator import add
from typing import Annotated, Any, Literal, TypedDict


TaskType = Literal[
    "data_profile",
    "rag_question",
    "sql_analysis",
    "chart_analysis",
    "statistical_analysis",
    "causal_analysis",
    "mixed_analysis",
    "unsupported",
]


WorkflowStatus = Literal[
    "initialized",
    "routing",
    "planning",
    "tool_decision",
    "tool_validation",
    "tool_execution",
    "checking_result",
    "reporting",
    "completed",
    "failed",
    "awaiting_approval",
    "blocked",
    "causal_execution",
]


class AgentState(TypedDict, total=False):
    """
    CausaLens LangGraph 工作流的共享状态。

    total=False 表示并非所有字段都必须在初始状态中出现。
    各个节点可以在工作流执行过程中逐步补充字段。
    """

    # =========================
    # 用户输入
    # =========================

    dataset_path: str
    question: str

    # =========================
    # 任务理解和规划
    # =========================

    task_type: TaskType

    # planner_node 通常一次性生成完整计划，
    # 因此这里使用普通 list，不设置追加 reducer。
    plan: list[str]

    # 完成当前问题所需的工具名称。
    # 例如：
    # ["retrieve_docs", "run_readonly_sql"]
    required_tools: list[str]

    workflow_status: WorkflowStatus

    # =========================
    # 模型消息历史
    # =========================

    # Annotated[..., add] 表示：
    # 新节点返回的消息会追加到已有消息后面，
    # 而不是覆盖整个 messages。
    messages: Annotated[list[dict[str, Any]], add]

    # =========================
    # 工具调用
    # =========================

    # tool_decision_node 生成的下一次待执行工具调用。
    # 一次只保存当前准备执行的工具。
    pending_tool_call: dict[str, Any] | None

    # 上一次候选工具调用没有通过验证的原因。
    validation_error: str

    # 当前工具已经连续验证失败的次数。
    validation_retry_count: int

    # 每个工具允许重新生成参数的最大次数。
    max_validation_retries: int

    # 保存完整工具调用历史。
    tool_calls: Annotated[list[dict[str, Any]], add]

    # 保存完整工具执行结果历史。
    tool_results: Annotated[list[dict[str, Any]], add]

    # =========================
    # 流程控制
    # =========================

    # 当前已经执行了多少次工具或工作流步骤。
    step_count: int

    # 最大允许步骤数，用于防止无限循环。
    max_steps: int

    # result_checker_node 可以通过它决定：
    # 是继续调用工具，还是进入最终报告。
    needs_more_tools: bool

    # 保存 result_checker_node 的判断原因，
    # 方便调试、测试和后续执行依据展示。
    result_check_reason: str

    # =========================
    # 最终输出与错误
    # =========================

    final_answer: str

    # 没有错误时为 None。
    error: str | None

    analysis_method: str | None

    treatment: str | None
    outcome: str | None
    confounders: list[str]

    causal_readiness: dict[str, Any]

    requires_approval: bool
    approved: bool | None


def create_initial_state(
    dataset_path: str,
    question: str,
    max_steps: int = 5,
    approved: bool | None = None,
) -> AgentState:
    """
    创建一次 CausaLens 工作流的初始状态。

    使用统一函数初始化，避免 CLI、测试和脚本分别构造
    不同格式的 State。
    """

    return {
        "dataset_path": dataset_path,
        "question": question,
        "plan": [],
        "required_tools": [],
        "workflow_status": "initialized",
        "messages": [],
        "pending_tool_call": None,
        "validation_error": "",
        "validation_retry_count": 0,
        "max_validation_retries": 2,
        "tool_calls": [],
        "tool_results": [],
        "step_count": 0,
        "max_steps": max_steps,
        "needs_more_tools": False,
        "result_check_reason": "",
        "final_answer": "",
        "error": None,
        "analysis_method": None,
        "treatment": None,
        "outcome": None,
        "confounders": [],
        "causal_readiness": {},
        "requires_approval": False,
        "approved":  approved,
    }