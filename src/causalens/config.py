from __future__ import annotations

import os
from dataclasses import dataclass


DEFAULT_BASE_URL = (
    "https://dashscope.aliyuncs.com/compatible-mode/v1"
)
DEFAULT_MODEL_NAME = "qwen3.8-flash"
DEFAULT_MAX_LLM_CALLS_PER_RUN = 6


def _read_positive_int(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    try:
        value = int(raw_value)
    except ValueError as exc:
        raise RuntimeError(
            f"{name} 必须是正整数。"
        ) from exc

    if value < 1:
        raise RuntimeError(
            f"{name} 必须是正整数。"
        )

    return value


def get_max_llm_calls_per_run() -> int:
    """读取调用预算，不要求在单元测试阶段提供 API Key。"""

    return _read_positive_int(
        "CAUSALENS_MAX_LLM_CALLS_PER_RUN",
        DEFAULT_MAX_LLM_CALLS_PER_RUN,
    )


def get_free_model_candidates() -> tuple[str, ...]:
    """
    读取用户已在百炼控制台确认开启“免费额度用完即停”的模型。

    FREE_MODEL_CANDIDATES 是逗号分隔的有序列表。为了兼容旧配置，
    未设置该变量时继续读取 MODEL_NAME；两者都未设置时使用默认模型。
    """

    raw_candidates = os.getenv("FREE_MODEL_CANDIDATES")
    if raw_candidates:
        candidates = tuple(
            dict.fromkeys(
                item.strip()
                for item in raw_candidates.split(",")
                if item.strip()
            )
        )
    else:
        candidates = (
            os.getenv("MODEL_NAME", DEFAULT_MODEL_NAME).strip(),
        )

    if not candidates or any(not item for item in candidates):
        raise RuntimeError(
            "至少需要配置一个免费额度模型。"
        )

    return candidates


@dataclass(frozen=True)
class Settings:
    """CausaLens 运行配置。不会从文件中读取或保存 API Key。"""

    dashscope_api_key: str
    dashscope_base_url: str
    free_model_candidates: tuple[str, ...]
    max_llm_calls_per_run: int
    request_timeout_seconds: float

    @classmethod
    def from_env(cls) -> "Settings":
        api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError(
                "未找到 DASHSCOPE_API_KEY 环境变量，"
                "请检查系统环境变量配置。"
            )

        raw_timeout = os.getenv(
            "DASHSCOPE_TIMEOUT_SECONDS",
            "60",
        )
        try:
            timeout = float(raw_timeout)
        except ValueError as exc:
            raise RuntimeError(
                "DASHSCOPE_TIMEOUT_SECONDS 必须是正数。"
            ) from exc

        if timeout <= 0:
            raise RuntimeError(
                "DASHSCOPE_TIMEOUT_SECONDS 必须是正数。"
            )

        return cls(
            dashscope_api_key=api_key,
            dashscope_base_url=os.getenv(
                "DASHSCOPE_BASE_URL",
                DEFAULT_BASE_URL,
            ).strip(),
            free_model_candidates=(
                get_free_model_candidates()
            ),
            max_llm_calls_per_run=get_max_llm_calls_per_run(),
            request_timeout_seconds=timeout,
        )
