from __future__ import annotations

import json

from causalens.preflight import run_preflight


def test_preflight_keeps_offline_runner_available_without_api_key(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.delenv("FREE_MODEL_CANDIDATES", raising=False)
    monkeypatch.delenv("MODEL_NAME", raising=False)

    result = run_preflight(tmp_path)

    assert result["offline_contract_runner_ready"] is True
    assert result["natural_language_agent_ready"] is False
    assert result["network_calls"] == 0
    assert result["secrets_exposed"] is False


def test_preflight_never_serializes_api_key(tmp_path, monkeypatch) -> None:
    secret = "sk-private-do-not-print-this"
    monkeypatch.setenv("DASHSCOPE_API_KEY", secret)
    monkeypatch.setenv("FREE_MODEL_CANDIDATES", "qwen3.8-flash,kimi-k3")
    monkeypatch.setenv("CAUSALENS_MAX_LLM_CALLS_PER_RUN", "4")

    result = run_preflight(tmp_path)
    serialized = json.dumps(result, ensure_ascii=False)

    assert result["natural_language_agent_ready"] is True
    assert secret not in serialized
    free_check = next(
        check
        for check in result["checks"]
        if check["name"] == "free_model_configuration"
    )
    assert free_check["detail"]["candidate_count"] == 2
    assert free_check["detail"]["max_llm_calls_per_run"] == 4


def test_preflight_treats_example_key_as_unconfigured(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DASHSCOPE_API_KEY", "your_dashscope_api_key_here")

    result = run_preflight(tmp_path)

    assert result["natural_language_agent_ready"] is False
