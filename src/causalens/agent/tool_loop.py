import json
from typing import Any

from causalens.agent.qwen_client import create_chat_completion
from causalens.agent.tool_registry import TOOL_REGISTRY, TOOLS
from causalens.schemas import AgentResponse, ToolCall, ToolResult


def build_assistant_tool_message(assistant_message: Any) -> dict[str, Any]:
    """
    将 SDK 返回的 assistant 消息，转换成后续请求可使用的普通字典。
    只保留 Function Calling 所需的字段。
    """

    return {
        "role": "assistant",
        "content": assistant_message.content or "",
        "tool_calls": [
            {
                "id": tool_call.id,
                "type": tool_call.type,
                "function": {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments,
                },
            }
            for tool_call in assistant_message.tool_calls
        ],
    }


def run_tool_agent(question: str, dataset_path: str) -> AgentResponse:
    """
    执行一次完整的“模型 → 工具 → 模型”流程。
    """

    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "你是一个可靠的数据分析助手。"
                "当用户的问题涉及数据集的行数、列名、字段类型、缺失值、"
                "数值统计或平均值时，必须调用工具获取真实数据，不得编造。"
                "拿到工具结果后，请使用简洁中文回答用户。"
            ),
        },
        {
            "role": "user",
            "content": (
                f"数据集路径：{dataset_path}\n"
                f"用户问题：{question}"
            ),
        },
    ]

    # 第一轮：让模型决定是否调用工具。
    first_response = create_chat_completion(
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
    )

    assistant_message = first_response.choices[0].message

    # 模型没有调用工具时，直接返回它的文本回答。
    if not assistant_message.tool_calls:
        return AgentResponse(
            answer=assistant_message.content or "模型没有返回可用回答。",
            confidence="low",
        )

    # 把“模型请求调用工具”的消息加入对话历史。
    messages.append(build_assistant_tool_message(assistant_message))

    recorded_tool_calls: list[ToolCall] = []
    recorded_tool_results: list[ToolResult] = []

    # 第二步：逐个执行模型请求的工具。
    for raw_tool_call in assistant_message.tool_calls:
        tool_name = raw_tool_call.function.name

        try:
            arguments = json.loads(raw_tool_call.function.arguments)

            if not isinstance(arguments, dict):
                raise ValueError("工具参数必须是 JSON 对象。")

        except (json.JSONDecodeError, ValueError) as exc:
            tool_result = ToolResult(
                tool_name=tool_name,
                success=False,
                error=f"无法解析工具参数：{exc}",
            )

            arguments = {}

        else:
            recorded_tool_calls.append(
                ToolCall(
                    tool_name=tool_name,
                    arguments=arguments,
                )
            )

            tool_function = TOOL_REGISTRY.get(tool_name)

            if tool_function is None:
                tool_result = ToolResult(
                    tool_name=tool_name,
                    success=False,
                    error=f"未注册的工具：{tool_name}",
                )
            else:
                try:
                    tool_result = tool_function(**arguments)
                except Exception as exc:
                    tool_result = ToolResult(
                        tool_name=tool_name,
                        success=False,
                        error=f"工具执行异常：{exc}",
                    )

        recorded_tool_results.append(tool_result)

        # 将真实工具结果追加进消息历史，交回给模型。
        messages.append(
            {
                "role": "tool",
                "tool_call_id": raw_tool_call.id,
                "content": tool_result.model_dump_json(),
            }
        )

    # 第二轮：模型根据真实工具结果组织最终答案。
    final_response = create_chat_completion(
        messages=messages,
        tools=TOOLS,
        tool_choice="none",
    )

    final_answer = final_response.choices[0].message.content

    return AgentResponse(
        answer=final_answer or "模型没有返回最终答案。",
        tool_calls=recorded_tool_calls,
        tool_results=recorded_tool_results,
        confidence="high",
    )