"""Scoring strategies for the llm-quality plugin.

Three deterministic strategies (``exact_match``, ``substring_match``,
``f1_token``) score the model's text against a reference answer with no
external dependencies. A fourth strategy, ``judge_llm``, delegates to a
small judge model via a second :class:`ModelClient`; it returns 1.0 when
the judge replies ``"1"`` and 0.0 otherwise. Judge failures (network,
rate limit, parse error) are caught and scored 0.0 with a counter
incremented on the supplied :class:`ScoreContext` so the plugin can
surface the count in the envelope's metrics block.

All scorers share the same ``(ctx: ScoreContext) -> float`` signature.
The deterministic three ignore ``ctx.question`` and ``ctx.judge_client``;
the judge scorer uses both. This uniform shape lets the registry stay a
single dict and lets the plugin treat all strategies the same way.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from inferencebench.harness.client import ModelClient


JUDGE_PROMPT = (
    'You are a strict grader. Given a question, a model\'s answer, and the ground-truth answer,\n'
    'reply with just "1" if the model\'s answer is correct, otherwise "0". No explanation.\n'
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

    Existing deterministic scorers only read ``reference`` and ``hypothesis``;
    the judge scorer additionally reads ``question`` and ``judge_client``.
    ``judge_errors`` and ``judge_cost_usd`` are output channels — the judge
    scorer increments them in place so the plugin can include them in the
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
    return (
        1.0
        if ctx.hypothesis.strip().lower() == ctx.reference.strip().lower()
        else 0.0
    )


def substring_match(ctx: ScoreContext) -> float:
    """Return 1.0 iff ``reference`` appears (case-insensitively) in ``hypothesis``."""
    return (
        1.0 if ctx.reference.strip().lower() in ctx.hypothesis.lower() else 0.0
    )


def f1_token(ctx: ScoreContext) -> float:
    """Token-level F1 between whitespace-split tokens of hypothesis / reference."""
    pred_tokens = ctx.hypothesis.lower().split()
    ref_tokens = ctx.reference.lower().split()
    if not pred_tokens or not ref_tokens:
        return 0.0

    pred_counts: dict[str, int] = {}
    for tok in pred_tokens:
        pred_counts[tok] = pred_counts.get(tok, 0) + 1
    ref_counts: dict[str, int] = {}
    for tok in ref_tokens:
        ref_counts[tok] = ref_counts.get(tok, 0) + 1

    overlap = 0
    for tok, ref_n in ref_counts.items():
        overlap += min(ref_n, pred_counts.get(tok, 0))
    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2.0 * precision * recall / (precision + recall)


def judge_llm(ctx: ScoreContext) -> float:
    """LLM-as-judge: ask a small model to grade the response (binary).

    Returns 1.0 when the judge's first reply character is ``"1"``, otherwise
    0.0. The judge prompt asks for a single-character verdict so parsing is
    trivial. Any exception during the judge call (network, rate limit,
    malformed response) records the error message on ``ctx.judge_errors``
    and scores 0.0 — never propagates the exception.

    When the underlying judge call returns a non-zero ``cost_usd``, the
    value is appended to ``ctx.judge_cost_usd`` so the plugin can include
    judge cost in the envelope's cost-per-million-tokens calculation.
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
        result = ctx.judge_client.complete(
            prompt, stream=False, max_tokens=4, temperature=0.0
        )
    except Exception as exc:
        ctx.judge_errors.append(str(exc))
        return 0.0
    if result.cost_usd:
        ctx.judge_cost_usd.append(float(result.cost_usd))
    text = (result.text or "").lstrip()
    if not text:
        return 0.0
    return 1.0 if text[0] == "1" else 0.0


def score_with_judge(
    question: str,
    hypothesis: str,
    reference: str,
    *,
    judge_client: ModelClient,
) -> tuple[float, ScoreContext]:
    """Convenience wrapper: score a single question via the judge.

    Returns ``(score, populated_ctx)`` so callers can inspect both the
    binary verdict and the side-channel data the judge scorer mutates
    (errors, cost). Most callers in the plugin path call :func:`judge_llm`
    directly with their own :class:`ScoreContext`; this helper exists for
    test / interactive ergonomics.
    """
    ctx = ScoreContext(
        reference=reference,
        hypothesis=hypothesis,
        question=question,
        judge_client=judge_client,
    )
    score = judge_llm(ctx)
    return score, ctx


SCORERS: dict[str, Callable[[ScoreContext], float]] = {
    "exact_match": exact_match,
    "substring_match": substring_match,
    "f1_token": f1_token,
    "judge_llm": judge_llm,
}
