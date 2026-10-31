from types import SimpleNamespace

import pytest

from causalens.agent import qwen_client
from causalens.config import DEFAULT_MODEL_NAME, Settings


class FakeQuotaError(Exception):
    status_code = 403
    body = {"code": "AllocationQuota.FreeTierOnly"}


@pytest.mark.parametrize('model,planning,expected_switch', [
    ('qwen3.8-flash', True, True), ('qwen3.8-flash', False, False), ('other-model', True, False)])
def test_planning_generation_settings(monkeypatch, model, planning, expected_switch):
    monkeypatch.setenv('DASHSCOPE_API_KEY', 'test-key')
    monkeypatch.setenv('FREE_MODEL_CANDIDATES', model)
    captured = []
    client = FakeClient([])
    client.completions.create = lambda **kwargs: captured.append(kwargs) or object()
    monkeypatch.setattr(qwen_client, 'get_qwen_client', lambda settings=None: client)
    qwen_client.create_chat_completion([{'role':'user', 'content':'test'}], planning_mode=planning)
    assert ('extra_body' in captured[0]) == expected_switch
    assert captured[0].get('max_tokens') == (2048 if planning else None)
    if expected_switch:
        assert captured[0]['extra_body'] == {'enable_thinking': False}


class FakeCompletions:
    def __init__(self, outcomes):
        self.outcomes = iter(outcomes)
        self.models = []

    def create(self, **kwargs):
        self.models.append(kwargs["model"])
        outcome = next(self.outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class FakeClient:
    def __init__(self, outcomes):
        self.completions = FakeCompletions(outcomes)
        self.chat = SimpleNamespace(completions=self.completions)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_settings_uses_snapshot_model_by_default(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
    monkeypatch.delenv("MODEL_NAME", raising=False)
    monkeypatch.delenv("FREE_MODEL_CANDIDATES", raising=False)

    settings = Settings.from_env()

    assert settings.free_model_candidates == (DEFAULT_MODEL_NAME,)


def test_rotates_only_after_free_quota_exhaustion(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
    monkeypatch.setenv(
        "FREE_MODEL_CANDIDATES",
        "qwen3.7-flash-2026-07-15,qwen3.8-flash",
    )
    expected = object()
    client = FakeClient([FakeQuotaError("free quota"), expected])
    monkeypatch.setattr(
        qwen_client,
        "get_qwen_client",
        lambda settings=None: client,
    )

    result = qwen_client.create_chat_completion(
        messages=[{"role": "user", "content": "test"}],
    )

    assert result is expected
    assert client.completions.models == [
        "qwen3.7-flash-2026-07-15",
        "qwen3.8-flash",
    ]


def test_does_not_rotate_on_non_quota_error(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
    monkeypatch.setenv(
        "FREE_MODEL_CANDIDATES",
        "first-free,second-free",
    )
    client = FakeClient([RuntimeError("network unavailable")])
    monkeypatch.setattr(
        qwen_client,
        "get_qwen_client",
        lambda settings=None: client,
    )

    with pytest.raises(RuntimeError, match="network unavailable"):
        qwen_client.create_chat_completion(
            messages=[{"role": "user", "content": "test"}],
        )

    assert client.completions.models == ["first-free"]


def test_call_budget_stops_before_extra_request(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
    client = FakeClient([object(), object()])
    monkeypatch.setattr(
        qwen_client,
        "get_qwen_client",
        lambda settings=None: client,
    )

    with qwen_client.llm_call_budget(1):
        qwen_client.create_chat_completion(
            messages=[{"role": "user", "content": "first"}],
        )
        with pytest.raises(qwen_client.LLMCallBudgetExceededError):
            qwen_client.create_chat_completion(
                messages=[{"role": "user", "content": "second"}],
            )

    assert len(client.completions.models) == 1
