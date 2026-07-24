from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph


class DemoState(TypedDict, total=False):
    """
    LangGraph 工作流共享的状态。
    """

    user_input: str

    # 决定 node_a 执行后，是否继续执行 node_b。
    should_run_node_b: bool

    node_a_result: str
    node_b_result: str


def node_a(state: DemoState) -> dict[str, str]:
    """
    第一个节点：读取用户输入并写入处理结果。
    """

    user_input = state.get("user_input", "")

    print("\n[node_a] 开始执行")
    print(f"[node_a] 收到用户输入：{user_input}")

    return {
        "node_a_result": f"node_a 已处理：{user_input}",
    }


def node_b(state: DemoState) -> dict[str, str]:
    """
    第二个节点：读取 node_a 写入的结果。
    """

    node_a_result = state.get("node_a_result", "")

    print("\n[node_b] 开始执行")
    print(f"[node_b] 读取 node_a_result：{node_a_result}")

    return {
        "node_b_result": f"node_b 已接收到上一步结果：{node_a_result}",
    }


def route_after_node_a(
    state: DemoState,
) -> Literal["run_node_b", "finish"]:
    """
    条件路由函数。

    它不是普通处理节点，而是负责决定下一步去哪。
    """

    should_run_node_b = state.get("should_run_node_b", False)

    print("\n[router] 正在判断下一步")

    if should_run_node_b:
        print("[router] should_run_node_b=True，继续执行 node_b")
        return "run_node_b"

    print("[router] should_run_node_b=False，直接结束")
    return "finish"


def build_demo_graph():
    """
    创建并编译带条件分支的 LangGraph。
    """

    graph_builder = StateGraph(DemoState)

    graph_builder.add_node("node_a", node_a)
    graph_builder.add_node("node_b", node_b)

    # 图从 node_a 开始。
    graph_builder.add_edge(START, "node_a")

    # node_a 执行结束后，不再固定进入 node_b。
    # LangGraph 会调用 route_after_node_a 判断下一步。
    graph_builder.add_conditional_edges(
        "node_a",
        route_after_node_a,
        {
            "run_node_b": "node_b",
            "finish": END,
        },
    )

    # 只有进入 node_b 后，才通过这条边结束。
    graph_builder.add_edge("node_b", END)

    return graph_builder.compile()


def run_case(
    graph,
    case_name: str,
    should_run_node_b: bool,
) -> None:
    """
    运行一个测试案例。
    """

    print("\n" + "=" * 60)
    print(f"运行案例：{case_name}")
    print("=" * 60)

    initial_state: DemoState = {
        "user_input": "请分析这个 CSV 数据集",
        "should_run_node_b": should_run_node_b,
    }

    final_state = graph.invoke(initial_state)

    print("\n工作流执行结束")
    print("最终 State：")

    for key, value in final_state.items():
        print(f"- {key}: {value}")


def main() -> None:
    graph = build_demo_graph()

    # 案例一：继续执行 node_b。
    run_case(
        graph=graph,
        case_name="执行 node_b",
        should_run_node_b=True,
    )

    # 案例二：跳过 node_b，直接结束。
    run_case(
        graph=graph,
        case_name="跳过 node_b",
        should_run_node_b=False,
    )


if __name__ == "__main__":
    main()