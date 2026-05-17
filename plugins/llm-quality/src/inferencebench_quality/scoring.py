"""Deterministic scoring strategies for the llm-quality plugin.

Three pure functions, each ``(prediction, reference) -> float`` in ``[0.0, 1.0]``.
No LLM-as-judge — that's deferred to a later revision. These strategies are
robust, cheap, and deterministic enough to exercise the full envelope-signing
pipeline end to end without external dependencies.
"""

from __future__ import annotations


def exact_match(prediction: str, reference: str) -> float:
    """Return 1.0 iff ``prediction`` equals ``reference`` after strip + lowercase.

    Whitespace at the edges and case differences are ignored — useful for
    short-answer formats ("Answer with just the number.") where the model's
    response may have trailing punctuation or wrapping whitespace.
    """
    return 1.0 if prediction.strip().lower() == reference.strip().lower() else 0.0


def substring_match(prediction: str, reference: str) -> float:
    """Return 1.0 iff ``reference`` appears (case-insensitively) in ``prediction``.

    Forgiving scorer for short factual recall: the model may produce
    "The capital of France is Paris." and still get credit against
    ``reference == "Paris"``. An empty reference is treated as a vacuous
    match and scores 1.0 — callers wanting strict behaviour should validate
    fixtures before calling.
    """
    return 1.0 if reference.strip().lower() in prediction.lower() else 0.0


def f1_token(prediction: str, reference: str) -> float:
    """Token-level F1 between whitespace-split tokens of ``prediction`` / ``reference``.

    Standard SQuAD-style F1: precision = |overlap| / |pred tokens|,
    recall = |overlap| / |ref tokens|, F1 = 2PR/(P+R). Case-insensitive,
    whitespace-tokenised; punctuation is treated as part of the surrounding
    token (the simplest defensible choice — sophisticated tokenisation is the
    judge model's job in the future revision).

    Returns 0.0 when either side has no tokens (avoids divide-by-zero and
    matches the convention that an empty prediction can never get credit).
    """
    pred_tokens = prediction.lower().split()
    ref_tokens = reference.lower().split()
    if not pred_tokens or not ref_tokens:
        return 0.0

    # Multiset overlap so repeated tokens count proportionally.
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


SCORERS = {
    "exact_match": exact_match,
    "substring_match": substring_match,
    "f1_token": f1_token,
}
