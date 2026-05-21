"""Scoring strategies for the vision-understanding plugin.

Three strategies share the same ``(ctx: ScoreContext) -> float`` signature:
``exact_match`` and ``substring_match`` are deterministic and ignore the
judge plumbing; ``judge_llm`` delegates to a small text-only judge model.
Patterns mirror :mod:`inferencebench_quality.scoring` so the plugin layer
above can stay symmetric across modalities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from inferencebench.harness.client import ModelClient


JUDGE_PROMPT = (
    "You are a strict grader for vision-language model answers. Given a question, "
    'the model\'s answer, and the ground-truth answer, reply with just "1" if '
    'the model\'s answer is correct, otherwise "0". No explanation.\n'
    "\n"
    "Question: {question}\n"
    "Model answer: {hypothesis}\n"
    "Ground truth: {reference}\n"
    "\n"
    "Score (1 or 0):"
)


@dataclass
class ScoreContext:
    """Per-question context passed to a scorer.

    Deterministic scorers only read ``reference`` and ``hypothesis``; the
    judge scorer additionally reads ``question`` and ``judge_client``.
    ``judge_errors`` and ``judge_cost_usd`` are output channels — the judge
    scorer mutates them in place so the plugin can include them in the
    envelope metrics without a second pass over results.
    """

    reference: str
    hypothesis: str
    question: str = ""
    judge_client: ModelClient | None = None
    # Output channels (mutated by the judge scorer):
    judge_errors: list[str] = field(default_factory=list)
    judge_cost_usd: list[float] = field(default_factory=list)


def exact_match(ctx: ScoreContext) -> float:
    """Return 1.0 iff ``hypothesis`` equals ``reference`` after strip + lowercase."""
    return 1.0 if ctx.hypothesis.strip().lower() == ctx.reference.strip().lower() else 0.0


def substring_match(ctx: ScoreContext) -> float:
    """Return 1.0 iff ``reference`` appears (case-insensitively) in ``hypothesis``."""
    return 1.0 if ctx.reference.strip().lower() in ctx.hypothesis.lower() else 0.0


def judge_llm(ctx: ScoreContext) -> float:
    """LLM-as-judge: ask a small model to grade the response (binary).

    Returns 1.0 when the judge's first reply character is ``"1"``, otherwise
    0.0. Any exception during the judge call records the error message on
    ``ctx.judge_errors`` and scores 0.0 — never propagates the exception.
    Non-zero ``cost_usd`` from the judge call is appended to
    ``ctx.judge_cost_usd`` for envelope-level cost accounting.
    """
    if ctx.judge_client is None:
        ctx.judge_errors.append("no judge_client configured")
        return 0.0
    prompt = JUDGE_PROMPT.format(
        question=ctx.question,
        hypothesis=ctx.hypothesis,
        reference=ctx.reference,
    )
    try:
        result = ctx.judge_client.complete(prompt, stream=False, max_tokens=4, temperature=0.0)
    except Exception as exc:  # judge failures must not blow up the run
        ctx.judge_errors.append(str(exc))
        return 0.0
    if result.cost_usd:
        ctx.judge_cost_usd.append(float(result.cost_usd))
    text = (result.text or "").lstrip()
    if not text:
        return 0.0
    return 1.0 if text[0] == "1" else 0.0


SCORERS: dict[str, Callable[[ScoreContext], float]] = {
    "exact_match": exact_match,
    "substring_match": substring_match,
    "judge_llm": judge_llm,
}
