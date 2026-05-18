"""Tests for the multi-turn persona-consistency scoring strategies.

The deterministic scorer (``persona_consistency``) takes a transcript and a
marker list and returns a :class:`PersonaConsistencyResult`. The judge
variant (``judge_llm_persona``) delegates to a mocked judge client and
returns a numeric grade in ``[0, 1]``.
"""

from __future__ import annotations

from typing import Any

from inferencebench.harness.client import CompletionResult
from inferencebench_quality.scoring import (
    PersonaConsistencyResult,
    judge_llm_persona,
    persona_consistency,
)


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
class _StubJudgeClient:
    """Minimal ``ModelClient`` stand-in returning a canned text reply."""

    def __init__(
        self, reply: str = "5", *, raises: Exception | None = None
    ) -> None:
        self.reply = reply
        self.raises = raises
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
        return CompletionResult(
            text=self.reply,
            tokens_in=40,
            tokens_out=1,
            ttft_ms=5.0,
            total_ms=10.0,
            tpot_ms=5.0,
            cost_usd=0.0,
            finish_reason="stop",
            token_source="mock",
        )


# --------------------------------------------------------------------------- #
# persona_consistency                                                         #
# --------------------------------------------------------------------------- #
def test_persona_consistency_all_markers_present_scores_one() -> None:
    turns = [
        ("What's 2+2?", "Arr matey, four it be on the high sea!"),
        ("Capital of France?", "Arr, 'tis Paris, sailed there once on a ship."),
        ("Pluto fact?", "Yarr, Pluto be a dwarf, says this old captain."),
        ("Goodbye!", "Farewell, matey! May the sea be kind."),
    ]
    result = persona_consistency(turns, markers=["arr", "matey", "sea", "ship", "captain"])
    assert isinstance(result, PersonaConsistencyResult)
    assert result.score == 1.0
    assert result.drift_first_miss_turn is None
    assert result.markers_present_per_turn == [True, True, True, True]


def test_persona_consistency_drifts_after_turn_two() -> None:
    turns = [
        ("Q1", "Arr matey, this be the answer!"),
        ("Q2", "Aye, the ship sails on."),
        ("Q3", "The answer is photosynthesis, plants absorb light."),
        ("Q4", "Pluto is a dwarf planet beyond Neptune."),
    ]
    result = persona_consistency(turns, markers=["arr", "matey", "ship", "sea"])
    assert result.score == 0.5
    assert result.drift_first_miss_turn == 2
    assert result.markers_present_per_turn == [True, True, False, False]


def test_persona_consistency_no_markers_anywhere_scores_zero() -> None:
    turns = [
        ("Q1", "Plain answer one."),
        ("Q2", "Plain answer two."),
        ("Q3", "Plain answer three."),
    ]
    result = persona_consistency(turns, markers=["arr", "matey"])
    assert result.score == 0.0
    assert result.drift_first_miss_turn == 0
    assert result.markers_present_per_turn == [False, False, False]


def test_persona_consistency_case_insensitive() -> None:
    """A marker 'Arr' should match a response containing 'arr' or 'ARR'."""
    turns = [
        ("Q1", "arr, what a fine day"),
        ("Q2", "ARR, on the high seas"),
    ]
    result = persona_consistency(turns, markers=["Arr"])
    assert result.score == 1.0
    assert result.drift_first_miss_turn is None


def test_persona_consistency_empty_markers_returns_zero() -> None:
    """Defensive: persona with no markers cannot be measured."""
    turns = [("Q1", "anything"), ("Q2", "more text")]
    result = persona_consistency(turns, markers=[])
    assert result.score == 0.0
    assert result.markers_present_per_turn == [False, False]


def test_persona_consistency_empty_turns_returns_zero() -> None:
    result = persona_consistency([], markers=["arr"])
    assert result.score == 0.0
    assert result.drift_first_miss_turn is None
    assert result.markers_present_per_turn == []


def test_persona_consistency_partial_miss_in_middle() -> None:
    """A pattern present-present-MISS-present-present.

    Score = 4/5, drift_first_miss_turn = 2 (the first miss).
    """
    turns = [
        ("Q1", "Arr 1"),
        ("Q2", "matey 2"),
        ("Q3", "plain 3"),
        ("Q4", "Arr 4"),
        ("Q5", "matey 5"),
    ]
    result = persona_consistency(turns, markers=["arr", "matey"])
    assert result.score == 0.8
    assert result.drift_first_miss_turn == 2


# --------------------------------------------------------------------------- #
# judge_llm_persona                                                           #
# --------------------------------------------------------------------------- #
def test_judge_llm_persona_returns_seven_tenths() -> None:
    judge = _StubJudgeClient(reply="7")
    score = judge_llm_persona(
        [("Q1", "Arr 1"), ("Q2", "matey 2")],
        system_prompt="be a pirate",
        judge_client=judge,  # type: ignore[arg-type]
    )
    assert score == 0.7
    assert len(judge.prompts) == 1
    assert "Turn 1 user: Q1" in judge.prompts[0]
    assert "Turn 2 assistant: matey 2" in judge.prompts[0]


def test_judge_llm_persona_clamps_above_ten() -> None:
    """A judge that replies '15' is clamped to 1.0."""
    judge = _StubJudgeClient(reply="15")
    score = judge_llm_persona(
        [("Q", "A")],
        system_prompt="persona",
        judge_client=judge,  # type: ignore[arg-type]
    )
    assert score == 1.0


def test_judge_llm_persona_zero_reply_scores_zero() -> None:
    judge = _StubJudgeClient(reply="0")
    score = judge_llm_persona(
        [("Q", "A")],
        system_prompt="persona",
        judge_client=judge,  # type: ignore[arg-type]
    )
    assert score == 0.0


def test_judge_llm_persona_handles_exception() -> None:
    """A judge that raises records the error and scores 0.0."""
    judge = _StubJudgeClient(raises=RuntimeError("network down"))
    errors: list[str] = []
    score = judge_llm_persona(
        [("Q", "A")],
        system_prompt="persona",
        judge_client=judge,  # type: ignore[arg-type]
        judge_errors=errors,
    )
    assert score == 0.0
    assert errors == ["network down"]


def test_judge_llm_persona_garbage_reply_scores_zero() -> None:
    judge = _StubJudgeClient(reply="not a number")
    score = judge_llm_persona(
        [("Q", "A")],
        system_prompt="persona",
        judge_client=judge,  # type: ignore[arg-type]
    )
    assert score == 0.0


def test_judge_llm_persona_no_client_scores_zero() -> None:
    errors: list[str] = []
    score = judge_llm_persona(
        [("Q", "A")],
        system_prompt="persona",
        judge_client=None,  # type: ignore[arg-type]
        judge_errors=errors,
    )
    assert score == 0.0
    assert errors == ["no judge_client configured"]
