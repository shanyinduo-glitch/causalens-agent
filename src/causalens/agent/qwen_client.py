import os

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
            "未找到 DASHSCOPE_API_KEY 环境变量，请先检查系统环境变量配置。"
        )

    return OpenAI(
        api_key=api_key,
        base_url=base_url,
    )


def request_tool_call(user_question: str):
    """向千问发送用户问题和工具说明，返回模型的第一轮响应。"""

    client = get_qwen_client()
    model_name = os.getenv("MODEL_NAME", DEFAULT_MODEL_NAME)

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

    completion = client.chat.completions.create(
        model=model_name,
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
    )

    return completion