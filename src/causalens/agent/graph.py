from __future__ import annotations

from typing import Any, Literal

from langgraph.graph import END, START, StateGraph

from causalens.agent.nodes import (
    intent_router_node,
    planner_node,
    reporter_node,
    result_checker_node,
    tool_decision_node,
    tool_executor_node,
    tool_validator_node,
    causal_readiness_node,
    causal_execution_node,
)
from causalens.agent.state import AgentState, create_initial_state


def route_after_intent(
    state: AgentState,
) -> Literal["planner", "reporter"]:
    """
    unsupported 请求直接进入 Reporter，
    其他任务进入 Planner。
    """

    task_type = state.get("task_type", "unsupported")

    print("\n[route_after_intent] 正在判断下一步")
    print(f"[route_after_intent] 当前任务类型：{task_type}")

    if task_type == "unsupported":
        print("[route_after_intent] 请求不受支持，直接进入 reporter")
        return "reporter"

    print("[route_after_intent] 请求受支持，进入 planner")
    return "planner"

def route_after_causal_readiness(
    state: AgentState,
) -> Literal[
    "causal_execution",
    "end",
]:
    """
    readiness 通过且已确认时执行 DML；
    其余情况结束本轮 Graph。
    """

    readiness = state.get(
        "causal_readiness",
        {},
    )
    approved = state.get("approved")
    workflow_status = state.get(
        "workflow_status"
    )

    print(
        "\n[route_after_causal_readiness] "
        "正在判断 readiness 后的下一步"
    )

    if (
        isinstance(readiness, dict)
        and readiness.get("ready") is True
        and approved is True
        and workflow_status
        == "causal_execution"
    ):
        print(
            "[route_after_causal_readiness] "
            "准备度通过且已确认，"
            "进入 causal_execution"
        )
        return "causal_execution"

    print(
        "[route_after_causal_readiness] "
        "当前不执行 DML，结束本轮 Graph"
    )
    return "end"

def route_after_planner(
    state: AgentState,
) -> Literal[
    "causal_readiness",
    "tool_decision",
]:
    """
    因果任务进入固定 readiness 节点；
    其他任务继续使用普通工具决策流程。
    """

    task_type = state.get(
        "task_type",
        "unsupported",
    )

    print(
        "\n[route_after_planner] "
        "正在判断计划后的下一步"
    )

    if task_type == "causal_analysis":
        print(
            "[route_after_planner] "
            "因果任务进入 causal_readiness"
        )
        return "causal_readiness"

    print(
        "[route_after_planner] "
        "普通任务进入 tool_decision"
    )
    return "tool_decision"

def route_after_causal_readiness(
    state: AgentState,
) -> Literal[
    "causal_execution",
    "end",
]:
    """
    readiness 通过且已确认时执行 DML；
    其余情况结束本轮 Graph。
    """

    readiness = state.get(
        "causal_readiness",
        {},
    )
    approved = state.get("approved")
    workflow_status = state.get(
        "workflow_status"
    )

    print(
        "\n[route_after_causal_readiness] "
        "正在判断 readiness 后的下一步"
    )

    if (
        isinstance(readiness, dict)
        and readiness.get("ready") is True
        and approved is True
        and workflow_status
        == "causal_execution"
    ):
        print(
            "[route_after_causal_readiness] "
            "准备度通过且已确认，"
            "进入 causal_execution"
        )
        return "causal_execution"

    print(
        "[route_after_causal_readiness] "
        "当前不执行 DML，结束本轮 Graph"
    )
    return "end"

def route_after_tool_decision(
    state: AgentState,
) -> Literal["tool_validator", "reporter"]:
    """
    工具决策成功后先进行参数验证；
    决策失败则进入 Reporter。
    """

    if state.get("error"):
        print(
            "\n[route_after_tool_decision] "
            "工具决策失败，进入 reporter"
        )
        return "reporter"

    if not state.get("pending_tool_call"):
        print(
            "\n[route_after_tool_decision] "
            "没有候选工具调用，进入 reporter"
        )
        return "reporter"

    print(
        "\n[route_after_tool_decision] "
        "工具决策成功，进入 tool_validator"
    )
    return "tool_validator"


def route_after_tool_validation(
    state: AgentState,
) -> Literal[
    "tool_executor",
    "tool_decision",
    "reporter",
]:
    """
    根据工具验证结果决定下一步。
    """

    if state.get("error"):
        print(
            "\n[route_after_tool_validation] "
            "达到验证重试上限，进入 reporter"
        )
        return "reporter"

    if state.get("pending_tool_call"):
        print(
            "\n[route_after_tool_validation] "
            "工具验证通过，进入 tool_executor"
        )
        return "tool_executor"

    print(
        "\n[route_after_tool_validation] "
        "工具验证失败，返回 tool_decision"
    )
    return "tool_decision"


def route_after_result_check(
    state: AgentState,
) -> Literal["tool_decision", "reporter"]:
    """
    Result Checker 判断还需要工具时继续循环，
    否则进入最终报告。
    """

    if state.get("error"):
        print(
            "\n[route_after_result_check] "
            "检测到错误，进入 reporter"
        )
        return "reporter"

    if state.get("needs_more_tools"):
        print(
            "\n[route_after_result_check] "
            "仍需工具，返回 tool_decision"
        )
        return "tool_decision"

    print(
        "\n[route_after_result_check] "
        "任务工具步骤已完成，进入 reporter"
    )
    return "reporter"


def build_causalens_graph():
    """
    创建并编译带工具验证和多工具循环的 CausaLens LangGraph。
    """

    graph_builder = StateGraph(AgentState)

    # 注册节点
    graph_builder.add_node(
        "intent_router",
        intent_router_node,
    )
    graph_builder.add_node(
        "planner",
        planner_node,
    )
    graph_builder.add_node(
        "causal_readiness",
        causal_readiness_node,
    )

    graph_builder.add_node(
        "causal_execution",
        causal_execution_node,
    )
    graph_builder.add_node(
        "tool_decision",
        tool_decision_node,
    )
    graph_builder.add_node(
        "tool_validator",
        tool_validator_node,
    )
    graph_builder.add_node(
        "tool_executor",
        tool_executor_node,
    )
    graph_builder.add_node(
        "result_checker",
        result_checker_node,
    )
    graph_builder.add_node(
        "reporter",
        reporter_node,
    )

    # 工作流入口
    graph_builder.add_edge(
        START,
        "intent_router",
    )

    # 意图路由
    graph_builder.add_conditional_edges(
        "intent_router",
        route_after_intent,
        {
            "planner": "planner",
            "reporter": "reporter",
        },
    )

    # 生成计划后进入工具决策
    # 因果任务进入固定 readiness 检查；
    # 其他任务继续进入普通工具决策。
    graph_builder.add_conditional_edges(
        "planner",
        route_after_planner,
        {
            "causal_readiness": (
                "causal_readiness"
            ),
            "tool_decision": "tool_decision",
        },
    )

    # readiness 通过且用户已确认时执行 DML；
    # 未确认或准备度未通过时结束本轮 Graph。
    graph_builder.add_conditional_edges(
        "causal_readiness",
        route_after_causal_readiness,
        {
            "causal_execution": (
                "causal_execution"
            ),
            "end": END,
        },
    )

    graph_builder.add_edge(
        "causal_execution",
        "reporter",
    )

    # 工具决策成功后先验证，不直接执行
    graph_builder.add_conditional_edges(
        "tool_decision",
        route_after_tool_decision,
        {
            "tool_validator": "tool_validator",
            "reporter": "reporter",
        },
    )

    # 验证成功则执行；验证失败则重新生成；
    # 达到重试上限则进入报告节点
    graph_builder.add_conditional_edges(
        "tool_validator",
        route_after_tool_validation,
        {
            "tool_executor": "tool_executor",
            "tool_decision": "tool_decision",
            "reporter": "reporter",
        },
    )

    # 工具执行后检查任务是否完成
    graph_builder.add_edge(
        "tool_executor",
        "result_checker",
    )

    # 还有工具则继续循环，否则生成最终报告
    graph_builder.add_conditional_edges(
        "result_checker",
        route_after_result_check,
        {
            "tool_decision": "tool_decision",
            "reporter": "reporter",
        },
    )

    # 工作流结束
    graph_builder.add_edge(
        "reporter",
        END,
    )

    return graph_builder.compile()


def run_causalens_graph(
    dataset_path: str,
    question: str,
    max_steps: int = 5,
    approved: bool | None = None,
) -> AgentState:
    """
    创建初始状态并运行 CausaLens 工作流。
    """

    graph = build_causalens_graph()

    initial_state = create_initial_state(
        dataset_path=dataset_path,
        question=question,
        max_steps=max_steps,
        approved=approved,
    )

    final_state: dict[str, Any] = graph.invoke(initial_state)

    return final_state