"""Shared fixtures for the vision-understanding plugin tests.

The plugin's network path runs through :class:`MultimodalClient.complete_multimodal`.
``make_mock_modelclient`` monkeypatches that method to return canned responses
keyed by ``(image_path, question)`` — tests never hit a real model.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from inferencebench.harness.client import CompletionResult
from inferencebench_vision.multimodal_client import MultimodalClient


@pytest.fixture
def make_mock_modelclient(monkeypatch: pytest.MonkeyPatch) -> Callable[..., None]:
    """Return a factory that swaps ``MultimodalClient.complete_multimodal``.

    Usage::

        def test_x(make_mock_modelclient):
            make_mock_modelclient(lambda image, question: "Paris")
            # ...subsequent complete_multimodal() calls return canned text.

    The factory takes a ``(image_path, question) -> text`` callable. Timings,
    tokens, and cost are deterministic (5 ms TTFT, 25 ms total, 16 tokens in/out,
    $0 cost).
    """

    def _factory(
        responder: Callable[[Path, str], str],
        *,
        ttft_ms: float = 5.0,
        total_ms: float = 25.0,
        tokens_in: int = 16,
        tokens_out: int = 16,
        cost_usd: float = 0.0,
    ) -> None:
        def _fake_complete(
            self: MultimodalClient,
            image_path: Path,
            question: str,
            *,
            max_tokens: int = 256,
            **extra: Any,
        ) -> CompletionResult:
            text = responder(image_path, question)
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

        monkeypatch.setattr(
            MultimodalClient, "complete_multimodal", _fake_complete
        )

    return _factory
