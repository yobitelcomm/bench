"""Tests for the LLM-as-judge scoring strategy.

The judge scorer takes a ``ScoreContext`` carrying a ``judge_client`` and
calls ``judge_client.complete(...)`` once per question, parsing the reply
as a binary verdict. These tests monkeypatch the judge client so no real
provider is ever contacted.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from inferencebench.envelope import generate_dev_keypair
from inferencebench.harness.client import CompletionResult, ModelClient
from inferencebench_quality import (
    EngineKind,
    LLMQualityPlugin,
    RunContext,
)
from inferencebench_quality.scoring import (
    ScoreContext,
    judge_llm,
    score_with_judge,
)


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
class _StubModelClient:
    """Minimal stand-in for ModelClient used by the judge scorer.

    Records the prompts it receives so tests can assert against them, and
    returns whatever ``responder(prompt)`` produces (or raises ``raises``
    if set, simulating a network / rate-limit failure).
    """

    def __init__(
        self,
        responder: Any = None,
        *,
        raises: Exception | None = None,
        cost_usd: float = 0.0,
    ) -> None:
        self.responder = responder
        self.raises = raises
        self.cost_usd = cost_usd
        self.prompts: list[str] = []

    def complete(
        self,
        prompt: str,
        *,
        max_tokens: int = 256,
        temperature: float = 0.0,
        stream: bool = True,
        system: str | None = None,
        **_: Any,
    ) -> CompletionResult:
        self.prompts.append(prompt)
        if self.raises is not None:
            raise self.raises
        text = self.responder(prompt) if callable(self.responder) else self.responder
        return CompletionResult(
            text=str(text),
            tokens_in=20,
            tokens_out=1,
            ttft_ms=5.0,
            total_ms=10.0,
            tpot_ms=5.0,
            cost_usd=self.cost_usd,
            finish_reason="stop",
            token_source="mock",
        )


# --------------------------------------------------------------------------- #
# judge_llm — unit tests                                                       #
# --------------------------------------------------------------------------- #
def test_score_with_judge_returns_one_when_judge_says_yes() -> None:
    judge = _StubModelClient(responder=lambda _p: "1")
    score, ctx = score_with_judge(
        "What is the capital of France?",
        "The answer is Paris.",
        "Paris",
        judge_client=judge,  # type: ignore[arg-type]
    )
    assert score == 1.0
    assert ctx.judge_errors == []
    assert "Paris" in judge.prompts[0]


def test_score_with_judge_returns_zero_when_judge_says_no() -> None:
    judge = _StubModelClient(responder=lambda _p: "0")
    score, _ctx = score_with_judge(
        "What is the capital of France?",
        "Lyon.",
        "Paris",
        judge_client=judge,  # type: ignore[arg-type]
    )
    assert score == 0.0


def test_judge_returns_zero_on_empty_reply() -> None:
    judge = _StubModelClient(responder=lambda _p: "")
    ctx = ScoreContext(
        reference="Paris",
        hypothesis="Lyon",
        question="?",
        judge_client=judge,  # type: ignore[arg-type]
    )
    assert judge_llm(ctx) == 0.0


def test_judge_returns_zero_on_freeform_text() -> None:
    judge = _StubModelClient(
        responder=lambda _p: "The model's answer looks correct"
    )
    ctx = ScoreContext(
        reference="Paris",
        hypothesis="Paris",
        question="?",
        judge_client=judge,  # type: ignore[arg-type]
    )
    # Reply doesn't start with "1" → 0.0.
    assert judge_llm(ctx) == 0.0


def test_judge_returns_zero_on_whitespace_only_reply() -> None:
    judge = _StubModelClient(responder=lambda _p: "   \n  ")
    ctx = ScoreContext(
        reference="Paris",
        hypothesis="Paris",
        question="?",
        judge_client=judge,  # type: ignore[arg-type]
    )
    assert judge_llm(ctx) == 0.0


def test_judge_handles_leading_whitespace_before_one() -> None:
    """Real judges sometimes prepend a space — strip leading whitespace before checking."""
    judge = _StubModelClient(responder=lambda _p: " 1")
    ctx = ScoreContext(
        reference="Paris",
        hypothesis="Paris",
        question="?",
        judge_client=judge,  # type: ignore[arg-type]
    )
    assert judge_llm(ctx) == 1.0


def test_judge_failure_scores_zero_and_records_error() -> None:
    judge = _StubModelClient(raises=RuntimeError("rate limit"))
    ctx = ScoreContext(
        reference="Paris",
        hypothesis="Paris",
        question="?",
        judge_client=judge,  # type: ignore[arg-type]
    )
    assert judge_llm(ctx) == 0.0
    assert len(ctx.judge_errors) == 1
    assert "rate limit" in ctx.judge_errors[0]


def test_judge_with_no_client_scores_zero_with_error() -> None:
    ctx = ScoreContext(reference="x", hypothesis="x", question="?")
    assert judge_llm(ctx) == 0.0
    assert ctx.judge_errors == ["no judge_client configured"]


def test_judge_records_cost_when_provider_reports_it() -> None:
    judge = _StubModelClient(responder=lambda _p: "1", cost_usd=0.000125)
    ctx = ScoreContext(
        reference="Paris",
        hypothesis="Paris",
        question="?",
        judge_client=judge,  # type: ignore[arg-type]
    )
    judge_llm(ctx)
    assert ctx.judge_cost_usd == [pytest.approx(0.000125)]


# --------------------------------------------------------------------------- #
# End-to-end: plugin.run with scoring: judge_llm                              #
# --------------------------------------------------------------------------- #
def _make_judge_client_factory(
    monkeypatch: pytest.MonkeyPatch, judge_replies: list[str]
) -> list[CompletionResult]:
    """Patch ModelClient.complete so the judge calls cycle through ``judge_replies``.

    The model-under-test always echoes the question (so the judge has
    something to grade); the judge model returns the next reply from
    ``judge_replies``. Returns the list of CompletionResults the patch
    produced (in call order) so tests can introspect them.
    """
    captured: list[CompletionResult] = []
    judge_idx = {"n": 0}

    judge_prompt_marker = "You are a strict grader."

    def _fake_complete(
        self: ModelClient,
        prompt: str,
        *,
        max_tokens: int = 256,
        temperature: float = 0.0,
        stream: bool = True,
        system: str | None = None,
        **_: Any,
    ) -> CompletionResult:
        if judge_prompt_marker in prompt:
            text = judge_replies[judge_idx["n"]]
            judge_idx["n"] += 1
        else:
            # Model-under-test: just echo the question back as the answer.
            text = prompt
        result = CompletionResult(
            text=text,
            tokens_in=10,
            tokens_out=4,
            ttft_ms=5.0,
            total_ms=25.0,
            tpot_ms=6.6,
            cost_usd=0.0,
            finish_reason="stop",
            token_source="mock",
        )
        captured.append(result)
        return result

    monkeypatch.setattr(ModelClient, "complete", _fake_complete)
    return captured


def test_end_to_end_judge_seven_of_ten_correct(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Judge replies "1" for the first 7 questions and "0" for the last 3."""
    judge_replies = ["1"] * 7 + ["0"] * 3
    _make_judge_client_factory(monkeypatch, judge_replies)

    private_key_path = tmp_path / "cosign.key"
    generate_dev_keypair(private_key_path)

    plugin = LLMQualityPlugin()
    spec = plugin.get_benchmark("llm.quality.factual-judged")
    ctx = RunContext(
        model_id="openai/mock-model",
        model_revision="abc1234",
        engine_kind=EngineKind.OPENAI,
        output_dir=tmp_path / "out",
        extra={
            "signing_mode": "dev",
            "dev_key_path": str(private_key_path),
        },
    )

    envelope = plugin.run(spec, ctx)

    assert envelope.metrics.get("n_judged") == 10.0
    assert envelope.metrics.get("judge_errors") == 0.0
    acc = envelope.metrics.get("accuracy")
    assert isinstance(acc, (int, float))
    assert float(acc) == pytest.approx(0.7)


def test_end_to_end_judge_respects_max_questions_cap(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``--judge-max-questions 3`` → only the first 3 questions are judged.

    Even though the fixture has 10 rows, only the first 3 contribute to
    ``accuracy`` / ``n_judged``. The remaining 7 are still sent to the
    model-under-test but skip the judge entirely (no error counter bump).
    """
    judge_replies = ["1", "1", "0"]  # only 3 calls should reach the judge
    _make_judge_client_factory(monkeypatch, judge_replies)

    private_key_path = tmp_path / "cosign.key"
    generate_dev_keypair(private_key_path)

    plugin = LLMQualityPlugin()
    spec = plugin.get_benchmark("llm.quality.factual-judged")
    ctx = RunContext(
        model_id="openai/mock-model",
        model_revision="abc1234",
        engine_kind=EngineKind.OPENAI,
        output_dir=tmp_path / "out",
        extra={
            "signing_mode": "dev",
            "dev_key_path": str(private_key_path),
            "judge_max_questions": 3,
        },
    )

    envelope = plugin.run(spec, ctx)
    assert envelope.metrics.get("n_judged") == 3.0
    assert envelope.metrics.get("judge_errors") == 0.0
    acc = envelope.metrics.get("accuracy")
    assert isinstance(acc, (int, float))
    # 2 of 3 judged correct → accuracy = 2/3.
    assert float(acc) == pytest.approx(2.0 / 3.0)


def test_end_to_end_judge_failure_counts_in_envelope(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A judge call that raises is scored 0.0 and counted in judge_errors."""
    judge_call_idx = {"n": 0}

    def _fake_complete(
        self: ModelClient,
        prompt: str,
        *,
        max_tokens: int = 256,
        temperature: float = 0.0,
        stream: bool = True,
        system: str | None = None,
        **_: Any,
    ) -> CompletionResult:
        if "You are a strict grader." in prompt:
            judge_call_idx["n"] += 1
            # Fail the 3rd judge call.
            if judge_call_idx["n"] == 3:
                msg = "simulated rate limit"
                raise RuntimeError(msg)
            return CompletionResult(
                text="1",
                tokens_in=20,
                tokens_out=1,
                ttft_ms=5.0,
                total_ms=10.0,
                tpot_ms=5.0,
                cost_usd=0.0,
                finish_reason="stop",
                token_source="mock",
            )
        return CompletionResult(
            text=prompt,
            tokens_in=10,
            tokens_out=4,
            ttft_ms=5.0,
            total_ms=25.0,
            tpot_ms=6.6,
            cost_usd=0.0,
            finish_reason="stop",
            token_source="mock",
        )

    monkeypatch.setattr(ModelClient, "complete", _fake_complete)

    private_key_path = tmp_path / "cosign.key"
    generate_dev_keypair(private_key_path)

    plugin = LLMQualityPlugin()
    spec = plugin.get_benchmark("llm.quality.factual-judged")
    ctx = RunContext(
        model_id="openai/mock-model",
        engine_kind=EngineKind.OPENAI,
        output_dir=tmp_path / "out",
        extra={
            "signing_mode": "dev",
            "dev_key_path": str(private_key_path),
        },
    )

    envelope = plugin.run(spec, ctx)
    assert envelope.metrics.get("n_judged") == 10.0
    assert envelope.metrics.get("judge_errors") == 1.0
    # 9 successful "1" replies + 1 failure scored 0 → accuracy = 0.9.
    acc = envelope.metrics.get("accuracy")
    assert isinstance(acc, (int, float))
    assert float(acc) == pytest.approx(0.9)
