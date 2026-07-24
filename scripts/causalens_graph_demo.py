from __future__ import annotations

from causalens.agent.graph import run_causalens_graph
from causalens.agent.state import AgentState


DATASET_PATH = "data/student_performance.csv"


def print_final_state(final_state: AgentState) -> None:
    """
    打印工作流最终状态中的关键字段。
    """

    print("\n" + "-" * 60)
    print("工作流最终状态")
    print("-" * 60)

    print(
        f"任务类型："
        f"{final_state.get('task_type', '未识别')}"
    )

    print(
        f"工作流状态："
        f"{final_state.get('workflow_status', '未知')}"
    )

    print("\n分析计划：")

    plan = final_state.get("plan", [])

    if plan:
        for index, step in enumerate(plan, start=1):
            print(f"{index}. {step}")
    else:
        print("暂无分析计划。")

    print("\n最终回答：")
    print(final_state.get("final_answer", "暂无最终回答。"))

    error = final_state.get("error")

    if error:
        print("\n错误信息：")
        print(error)


def run_case(
    case_name: str,
    question: str,
) -> None:
    """
    运行一个 CausaLens 工作流测试案例。
    """

    print("\n" + "=" * 70)
    print(f"测试案例：{case_name}")
    print(f"用户问题：{question}")
    print("=" * 70)

    final_state = run_causalens_graph(
        dataset_path=DATASET_PATH,
        question=question,
    )

    print_final_state(final_state)


def main() -> None:
    test_cases = [
        {
            "case_name": "数据概览任务",
            "question": "这个数据集有哪些字段？",
        },
        {
            "case_name": "文档检索任务",
            "question": "passed 字段是什么意思？",
        },
        {
            "case_name": "SQL 分析任务",
            "question": "不同专业的平均 score 是多少？",
        },
        {
            "case_name": "图表任务",
            "question": "请画出 study_hours 和 score 的散点图。",
        },
        {
            "case_name": "混合分析任务",
            "question": "请统计不同专业的平均 score，并画柱状图。",
        },
        {
            "case_name": "不支持的危险请求",
            "question": "请删除所有 score 小于 60 的学生。",
        },
    ]

    for test_case in test_cases:
        run_case(
            case_name=test_case["case_name"],
            question=test_case["question"],
        )


if __name__ == "__main__":
    main()