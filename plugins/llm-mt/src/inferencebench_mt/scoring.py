"""Deterministic scoring strategies for the llm-mt plugin.

Three pure functions, each ``(reference, hypothesis) -> float`` in ``[0.0, 1.0]``:

- :func:`chrf` — character n-gram F-score (the standard chrF metric).
- :func:`bleu_token` — corpus-free token BLEU with brevity penalty.
- :func:`exact_match` — strict strip + lowercase equality (mirrors llm-quality).

All three are higher-is-better — translation accuracy in :math:`[0, 1]`. No
external dependencies. Whitespace is normalised by collapsing runs before
character-n-gram extraction so chrF behaves the same on ``"hello  world"``
and ``"hello world"``.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable


def _char_ngrams(text: str, n: int) -> Counter[str]:
    """Return the multiset of character n-grams of length ``n`` in ``text``.

    Whitespace is collapsed (any run of whitespace becomes a single space) so
    formatting noise does not perturb the score. ``n`` must be at least 1.
    """
    normalised = " ".join(text.split())
    if len(normalised) < n:
        return Counter()
    return Counter(normalised[i : i + n] for i in range(len(normalised) - n + 1))


def chrf(reference: str, hypothesis: str, n: int = 6, beta: float = 2.0) -> float:
    """Character n-gram F-score (chrF).

    Collects character n-grams up to length ``n`` in both ``reference`` and
    ``hypothesis``, computes a single precision / recall / F_beta over the
    union of all n-gram orders, and returns the result in ``[0, 1]``. This
    is the simplest, hand-computable variant of the metric: order-uniform
    weighting, no separate word-vs-char split.

    Returns 1.0 when reference and hypothesis are identical (after
    whitespace normalisation) and 0.0 when no n-gram of any order overlaps.
    An empty reference and hypothesis match exactly (returns 1.0); only one
    side empty returns 0.0.
    """
    if n < 1:
        msg = "chrf requires n >= 1"
        raise ValueError(msg)
    ref_norm = " ".join(reference.split())
    hyp_norm = " ".join(hypothesis.split())
    if not ref_norm and not hyp_norm:
        return 1.0
    if not ref_norm or not hyp_norm:
        return 0.0

    total_match = 0
    total_hyp = 0
    total_ref = 0
    for order in range(1, n + 1):
        ref_ngrams = _char_ngrams(reference, order)
        hyp_ngrams = _char_ngrams(hypothesis, order)
        if not ref_ngrams or not hyp_ngrams:
            continue
        overlap = sum((ref_ngrams & hyp_ngrams).values())
        total_match += overlap
        total_hyp += sum(hyp_ngrams.values())
        total_ref += sum(ref_ngrams.values())

    if total_hyp == 0 or total_ref == 0 or total_match == 0:
        return 0.0
    precision = total_match / total_hyp
    recall = total_match / total_ref
    beta_sq = beta * beta
    denom = beta_sq * precision + recall
    if denom == 0:
        return 0.0
    return (1.0 + beta_sq) * precision * recall / denom


def bleu_token(reference: str, hypothesis: str, max_n: int = 4) -> float:
    """Simple corpus-free BLEU over whitespace-split tokens.

    Computes modified n-gram precisions for ``n`` in ``1..max_n``, takes
    their geometric mean, and multiplies by the standard brevity penalty
    ``exp(min(0, 1 - r/c))``. Returns 0.0 when any n-gram order has zero
    precision (the canonical BLEU behaviour — no smoothing here, since the
    metric is informational only for the skeleton). Result is in ``[0, 1]``.
    """
    if max_n < 1:
        msg = "bleu_token requires max_n >= 1"
        raise ValueError(msg)
    hyp_tokens = hypothesis.split()
    ref_tokens = reference.split()
    if not hyp_tokens or not ref_tokens:
        return 0.0

    precisions: list[float] = []
    for n in range(1, max_n + 1):
        if len(hyp_tokens) < n or len(ref_tokens) < n:
            return 0.0
        hyp_ngrams = Counter(
            tuple(hyp_tokens[i : i + n]) for i in range(len(hyp_tokens) - n + 1)
        )
        ref_ngrams = Counter(
            tuple(ref_tokens[i : i + n]) for i in range(len(ref_tokens) - n + 1)
        )
        # Modified count: cap each n-gram by its reference count.
        clipped = sum(min(c, ref_ngrams[g]) for g, c in hyp_ngrams.items())
        total = sum(hyp_ngrams.values())
        if total == 0 or clipped == 0:
            return 0.0
        precisions.append(clipped / total)

    # Geometric mean of precisions.
    log_sum = sum(math.log(p) for p in precisions)
    geo_mean = math.exp(log_sum / len(precisions))

    c = len(hyp_tokens)
    r = len(ref_tokens)
    brevity = 1.0 if c > r else math.exp(1.0 - r / c)
    return brevity * geo_mean


def exact_match(reference: str, hypothesis: str) -> float:
    """Return 1.0 iff ``hypothesis`` equals ``reference`` after strip + lowercase."""
    return 1.0 if hypothesis.strip().lower() == reference.strip().lower() else 0.0


SCORERS: dict[str, Callable[[str, str], float]] = {
    "chrf": chrf,
    "bleu_token": bleu_token,
    "exact_match": exact_match,
}
