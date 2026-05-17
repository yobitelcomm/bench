"""Shared fixtures for the llm-quality plugin tests.

We never want to hit a real provider in unit tests, so ``_make_mock_modelclient``
monkeypatches :class:`inferencebench.harness.client.ModelClient.complete` with
a caller-supplied callable. The callable receives the prompt string and returns
the raw text the mock should emit; timings, tokens, and cost are filled in
with deterministic placeholders.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from inferencebench.harness.client import CompletionResult, ModelClient


@pytest.fixture
def make_mock_modelclient(monkeypatch: pytest.MonkeyPatch) -> Callable[..., None]:
    """Return a factory that swaps ``ModelClient.complete`` for a canned response.

    Usage::

        def test_x(make_mock_modelclient):
            make_mock_modelclient(lambda prompt: "Paris")
            # ...subsequent ModelClient.complete() calls return canned text.

    The factory takes a ``prompt -> text`` callable. Timings, tokens, and cost
    are deterministic (5 ms TTFT, 25 ms total, 16 tokens in/out, $0 cost).
    """

    def _factory(
        responder: Callable[[str], str],
        *,
        ttft_ms: float = 5.0,
        total_ms: float = 25.0,
        tokens_in: int = 16,
        tokens_out: int = 16,
        cost_usd: float = 0.0,
    ) -> None:
        def _fake_complete(
            self: ModelClient,
            prompt: str,
            *,
            max_tokens: int = 256,
            temperature: float = 0.0,
            stream: bool = True,
            system: str | None = None,
            **extra: Any,
        ) -> CompletionResult:
            text = responder(prompt)
            tpot = (total_ms - ttft_ms) / max(tokens_out - 1, 1)
            return CompletionResult(
                text=text,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                ttft_ms=ttft_ms,
                total_ms=total_ms,
                tpot_ms=tpot,
                cost_usd=cost_usd,
                finish_reason="stop",
                token_source="mock",
            )

        monkeypatch.setattr(ModelClient, "complete", _fake_complete)

    return _factory
