"""Deterministic IR scoring strategies for the embeddings-retrieval plugin.

Three pure functions, each ``(ranking, relevant, k) -> float`` in ``[0.0, 1.0]``.
All higher-is-better — 1.0 means the ranking placed every relevant doc at the
top k, 0.0 means none of the relevant docs appear in the top k.

No real embedding model is invoked; these are standard IR metrics on
already-produced rank lists.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence

MetricFn = Callable[[Sequence[str], Iterable[str], int], float]


def _topk(ranking: Sequence[str], k: int) -> list[str]:
    """Slice the top-k of a ranking; tolerate ``k`` exceeding the ranking length."""
    if k <= 0:
        return []
    return list(ranking[:k])


def recall_at_k(
    ranking: Sequence[str],
    relevant: Iterable[str],
    k: int,
) -> float:
    """Fraction of relevant docs that appear in the top-k of the ranking.

    Standard recall@k: |{relevant ∩ top-k}| / |relevant|. Returns 0.0 when
    the relevant set is empty (vacuously no docs to recall). Capped at 1.0
    by construction.
    """
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    top = set(_topk(ranking, k))
    hits = len(top & relevant_set)
    return hits / len(relevant_set)


def mrr_at_k(
    ranking: Sequence[str],
    relevant: Iterable[str],
    k: int,
) -> float:
    """Reciprocal rank of the first relevant doc within top-k, or 0.0 if none.

    Strictly speaking "MRR" is the mean of reciprocal ranks across queries,
    but per-query the value used in that mean is ``1 / rank_of_first_hit``,
    and that is what this function returns. The caller aggregates by taking
    the mean across queries.
    """
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    for idx, doc_id in enumerate(_topk(ranking, k), start=1):
        if doc_id in relevant_set:
            return 1.0 / idx
    return 0.0


def ndcg_at_k(
    ranking: Sequence[str],
    relevant: Iterable[str],
    k: int,
) -> float:
    """Normalised discounted cumulative gain @ k with binary relevance.

    Standard binary-relevance nDCG: ``DCG@k = sum_{i=1..k} rel_i / log2(i+1)``
    where ``rel_i`` is 1 if the i-th ranked doc is relevant else 0. The
    ideal DCG is the same sum when the top ``min(k, |relevant|)`` slots are
    all relevant. nDCG = DCG / IDCG. Returns 0.0 for an empty relevant set.
    """
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    top = _topk(ranking, k)
    dcg = 0.0
    for i, doc_id in enumerate(top, start=1):
        if doc_id in relevant_set:
            dcg += 1.0 / math.log2(i + 1)
    ideal_hits = min(k, len(relevant_set))
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


# Map metric-name -> ``(k, scorer)``. Scorer signature: (ranking, relevant, k).
METRICS: dict[str, tuple[int, MetricFn]] = {
    "recall_at_5": (5, recall_at_k),
    "mrr_at_10": (10, mrr_at_k),
    "ndcg_at_10": (10, ndcg_at_k),
}
