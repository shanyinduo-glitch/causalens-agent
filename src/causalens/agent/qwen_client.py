from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from openai import OpenAI

from causalens.config import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL_NAME,
    Settings,
    get_free_model_candidates,
)


_remaining_calls: ContextVar[int | None] = ContextVar(
    "causalens_remaining_llm_calls",
    default=None,
)


class FreeModelQuotaExhaustedError(RuntimeError):
    """所有配置的免费额度模型均已耗尽。"""


class LLMCallBudgetExceededError(RuntimeError):
    """单次工作流允许的模型调用次数已经耗尽。"""


@contextmanager
def llm_call_budget(limit: int):
    """为一次工作流设置独立的 LLM 调用次数上限。"""

    if limit < 1:
        raise ValueError("LLM 调用预算必须至少为 1。")
    token = _remaining_calls.set(limit)
    try:
        yield
    finally:
        _remaining_calls.reset(token)


def _consume_call_budget() -> None:
    remaining = _remaining_calls.get()
    if remaining is None:
        return
    if remaining <= 0:
        raise LLMCallBudgetExceededError(
            "本次分析已达到模型调用次数上限，"
            "为避免继续消耗免费额度，工作流已停止。"
        )
    _remaining_calls.set(remaining - 1)


def _is_free_quota_exhausted(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    body = getattr(exc, "body", None)
    searchable = f"{exc} {body}".lower()
    quota_markers = (
        "allocationquota.freetieronly",
        "free tier",
        "free quota",
        "免费额度",
    )
    return status_code == 403 and any(
        marker in searchable for marker in quota_markers
    )


def get_qwen_client(settings: Settings | None = None) -> OpenAI:
    """创建并返回千问 OpenAI 兼容客户端。"""

    resolved = settings or Settings.from_env()

    return OpenAI(
        api_key=resolved.dashscope_api_key,
        base_url=resolved.dashscope_base_url,
        timeout=resolved.request_timeout_seconds,
        # SDK 隐式重试可能重复消耗 Token。重试由上层显式决定。
        max_retries=0,
    )


def get_model_name() -> str:
    """返回免费额度候选列表中的首选模型。"""

    return get_free_model_candidates()[0]


def create_chat_completion(
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
    response_format: dict[str, Any] | None = None,
    planning_mode: bool = False,
):
    """
    调用已确认具有免费额度保护的模型。

    只有服务明确返回免费额度耗尽错误时才尝试下一个候选模型；
    网络错误、参数错误和服务端错误都不会触发轮换。
    """

    settings = Settings.from_env()

    request_kwargs: dict[str, Any] = {
        "messages": messages,
    }

    if tools is not None:
        request_kwargs["tools"] = tools

    if tool_choice is not None:
        request_kwargs["tool_choice"] = tool_choice

    if response_format is not None:
        request_kwargs["response_format"] = response_format
    if planning_mode:
        request_kwargs['max_tokens'] = 2048

    exhausted_models: list[str] = []
    with get_qwen_client(settings) as client:
        for model_name in settings.free_model_candidates:
            model_kwargs = dict(request_kwargs)
            # Only pass the provider-specific switch to the model verified in live probes.
            if planning_mode and model_name == 'qwen3.8-flash':
                model_kwargs['extra_body'] = {'enable_thinking': False}
            _consume_call_budget()
            try:
                return client.chat.completions.create(
                    model=model_name,
                    **model_kwargs,
                )
            except Exception as exc:
                if not _is_free_quota_exhausted(exc):
                    raise
                exhausted_models.append(model_name)

    raise FreeModelQuotaExhaustedError(
        "已配置模型的免费额度均已耗尽，未切换到任何付费模型："
        + ", ".join(exhausted_models)
    )


def request_tool_call(user_question: str):
    """保留 Day 4 的演示函数：请求模型选择工具。"""

    # 延迟导入，避免普通模型配置读取加载完整数据工具依赖。
    from causalens.agent.tool_registry import TOOLS

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
