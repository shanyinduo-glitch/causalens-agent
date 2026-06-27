from causalens.agent.tool_loop import run_tool_agent


QUESTIONS = [
    "这个数据集有多少行？",
    "有哪些字段？",
    "哪个字段缺失值最多？",
    "score 的平均值是多少？",
]


def main() -> None:
    dataset_path = "data/sample.csv"

    for index, question in enumerate(QUESTIONS, start=1):
        print("=" * 60)
        print(f"问题 {index}：{question}")

        result = run_tool_agent(
            question=question,
            dataset_path=dataset_path,
        )

        print("\n最终回答：")
        print(result.answer)

        print("\n调用的工具：")
        for tool_call in result.tool_calls:
            print(
                f"- 工具名：{tool_call.tool_name}"
                f"，参数：{tool_call.arguments}"
            )

        print("\n工具是否成功：")
        for tool_result in result.tool_results:
            print(
                f"- 工具名：{tool_result.tool_name}"
                f"，success={tool_result.success}"
            )

        print()


if __name__ == "__main__":
    main()