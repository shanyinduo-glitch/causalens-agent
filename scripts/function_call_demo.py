import json

from causalens.agent.qwen_client import request_tool_call


def main() -> None:
    user_question = "请查看 data/sample.csv 的基本情况。"

    print("=== 用户问题 ===")
    print(user_question)

    print("\n=== 正在请求千问选择工具 ===")
    completion = request_tool_call(user_question)

    message = completion.choices[0].message

    print("\n=== 千问第一轮回复 ===")
    print(message.model_dump())

    if not message.tool_calls:
        print("\n模型本轮没有调用工具。")
        print("普通回复：", message.content)
        return

    print("\n=== 解析出的工具调用 ===")

    for tool_call in message.tool_calls:
        tool_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)

        print(
            {
                "name": tool_name,
                "arguments": arguments,
            }
        )


if __name__ == "__main__":
    main()