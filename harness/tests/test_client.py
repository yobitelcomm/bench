"""Tests for the ModelClient.

Real LiteLLM calls are tested in `@pytest.mark.paid` tests against a CI budget.
Here we exercise the streaming/blocking branches with mocks.
"""

from __future__ import annotations

import sys
import types
from collections.abc import Iterator
from typing import Any

import pytest

from inferencebench.harness.client import (
    ClientError,
    CompletionResult,
    ModelClient,
    detect_endpoint_health,
    env_api_key,
)


# --------------------------------------------------------------------------- #
# Mock LiteLLM injection                                                      #
# --------------------------------------------------------------------------- #
class _MockDelta:
    def __init__(self, content: str | None) -> None:
        self.content = content


class _MockChoice:
    def __init__(self, content: str, finish: str | None = None) -> None:
        self.delta = _MockDelta(content)
        self.message = types.SimpleNamespace(content=content)
        self.finish_reason = finish


class _MockUsage:
    def __init__(self, p: int, c: int) -> None:
        self.prompt_tokens = p
        self.completion_tokens = c


class _MockResponse:
    def __init__(self, choices: list[_MockChoice], usage: _MockUsage | None = None) -> None:
        self.choices = choices
        self.usage = usage
        self._hidden_params: dict[str, Any] = {}


def _make_stream(
    chunks: list[str], *, final_finish: str = "stop", usage: _MockUsage | None = None
) -> Iterator[_MockResponse]:
    """Yield a sequence of mock streaming chunks, attaching usage on the last."""
    for i, c in enumerate(chunks):
        is_last = i == len(chunks) - 1
        resp = _MockResponse([_MockChoice(c, finish=final_finish if is_last else None)])
        if is_last and usage:
            resp.usage = usage
        yield resp


@pytest.fixture
def mock_litellm(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Install a fake litellm module into sys.modules for the duration of the test."""
    fake = types.ModuleType("litellm")
    monkeypatch.setitem(sys.modules, "litellm", fake)
    return fake


# --------------------------------------------------------------------------- #
# Streaming path                                                              #
# --------------------------------------------------------------------------- #
def test_streaming_aggregates_chunks(mock_litellm: types.ModuleType) -> None:
    chunks = ["Hello, ", "world. ", "How can ", "I help?"]
    usage = _MockUsage(p=10, c=5)
    mock_litellm.completion = lambda **kw: _make_stream(chunks, usage=usage)

    client = ModelClient(model="openai/gpt-4o-mini")
    result = client.complete("Test prompt", max_tokens=50, stream=True)

    assert isinstance(result, CompletionResult)
    assert result.text == "Hello, world. How can I help?"
    assert result.tokens_in == 10
    assert result.tokens_out == 5
    assert result.token_source == "provider"
    assert result.finish_reason == "stop"
    assert result.ttft_ms >= 0
    assert result.total_ms >= result.ttft_ms


def test_streaming_falls_back_to_tiktoken_when_no_usage(
    mock_litellm: types.ModuleType,
) -> None:
    chunks = ["Hello, world."]
    mock_litellm.completion = lambda **kw: _make_stream(chunks, usage=None)

    client = ModelClient(model="openai/gpt-4o-mini")
    result = client.complete("Test prompt", stream=True)
    assert result.token_source in ("tiktoken", "approx-words")
    assert result.tokens_in > 0
    assert result.tokens_out > 0


def test_streaming_propagates_provider_cost(mock_litellm: types.ModuleType) -> None:
    chunks = ["abc"]

    def _stream(**kw: Any) -> Iterator[_MockResponse]:
        for i, c in enumerate(chunks):
            r = _MockResponse([_MockChoice(c, finish="stop" if i == len(chunks) - 1 else None)])
            r._hidden_params = {"response_cost": 0.000123}
            yield r

    mock_litellm.completion = _stream
    client = ModelClient(model="openai/gpt-4o-mini")
    result = client.complete("p", stream=True)
    assert result.cost_usd == pytest.approx(0.000123)


# --------------------------------------------------------------------------- #
# Blocking path                                                               #
# --------------------------------------------------------------------------- #
def test_blocking_returns_full_text(mock_litellm: types.ModuleType) -> None:
    response = _MockResponse(
        choices=[_MockChoice("Hello, world.", finish="stop")],
        usage=_MockUsage(p=4, c=3),
    )
    mock_litellm.completion = lambda **kw: response

    client = ModelClient(model="openai/gpt-4o-mini")
    result = client.complete("Test prompt", stream=False)

    assert result.text == "Hello, world."
    assert result.tokens_in == 4
    assert result.tokens_out == 3
    assert result.token_source == "provider"
    assert result.ttft_ms == result.total_ms  # blocking → TTFT == total


# --------------------------------------------------------------------------- #
# Error paths                                                                 #
# --------------------------------------------------------------------------- #
def test_completion_raises_client_error_on_provider_failure(
    mock_litellm: types.ModuleType,
) -> None:
    def _raise(**kw: Any) -> Iterator[_MockResponse]:
        raise RuntimeError("network broke")

    mock_litellm.completion = _raise

    client = ModelClient(model="openai/gpt-4o-mini")
    with pytest.raises(ClientError, match="completion failed"):
        client.complete("p", stream=True)


def test_missing_litellm_raises_clienterror(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "litellm", None)
    client = ModelClient(model="openai/gpt-4o-mini")
    with pytest.raises(ClientError, match="litellm is required"):
        client.complete("p")


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def test_env_api_key_returns_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert env_api_key("openai") == "sk-test"


def test_env_api_key_returns_none_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert env_api_key("openai") is None


def test_env_api_key_unknown_provider() -> None:
    assert env_api_key("nonexistent-vendor") is None


def test_detect_endpoint_health_returns_bool_for_unreachable() -> None:
    # Unreachable URL → False (no exception)
    assert detect_endpoint_health("http://127.0.0.1:1", timeout_s=1.0) is False
