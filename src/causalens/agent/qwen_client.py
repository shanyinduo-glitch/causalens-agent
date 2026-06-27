import os
from typing import Any

from openai import OpenAI

from causalens.agent.tool_registry import TOOLS


DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL_NAME = "qwen3.7-plus"


def get_qwen_client() -> OpenAI:
    """创建并返回千问 OpenAI 兼容客户端。"""

    api_key = os.getenv("DASHSCOPE_API_KEY")
    base_url = os.getenv("DASHSCOPE_BASE_URL", DEFAULT_BASE_URL)

    if not api_key:
        raise RuntimeError(
            "未找到 DASHSCOPE_API_KEY 环境变量，请检查系统环境变量配置。"
        )

    return OpenAI(
        api_key=api_key,
        base_url=base_url,
    )


def get_model_name() -> str:
    """读取当前要使用的模型名称。"""

    return os.getenv("MODEL_NAME", DEFAULT_MODEL_NAME)


def create_chat_completion(
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | None = None,
):
    """向千问发送一次对话请求。"""

    request_kwargs: dict[str, Any] = {
        "model": get_model_name(),
        "messages": messages,
    }

    if tools is not None:
        request_kwargs["tools"] = tools

    if tool_choice is not None:
        request_kwargs["tool_choice"] = tool_choice

    client = get_qwen_client()

    return client.chat.completions.create(**request_kwargs)


def request_tool_call(user_question: str):
    """保留 Day 4 的演示函数：请求模型选择工具。"""

    messages = [
        {
            "role": "system",
            "content": (
                "你是一个 CSV 数据分析助手。"
                "当用户询问 CSV 数据集的基本情况时，"
                "请调用 profile_dataset 工具。"
            ),
        },
        {
            "role": "user",
            "content": user_question,
        },
    ]

    return create_chat_completion(
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
    )