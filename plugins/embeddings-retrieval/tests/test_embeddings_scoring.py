"""Tests for recall@k / MRR@k / nDCG@k retrieval scorers."""

from __future__ import annotations

import math

import pytest

from inferencebench_embeddings.scoring import (
    METRICS,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
)


# --------------------------------------------------------------------------- #
# recall@k                                                                    #
# --------------------------------------------------------------------------- #
def test_recall_perfect_when_all_relevant_in_topk() -> None:
    ranking = ["doc-1", "doc-2", "doc-3", "doc-4", "doc-5"]
    relevant = ["doc-1", "doc-3"]
    assert recall_at_k(ranking, relevant, 5) == 1.0


def test_recall_partial_hit() -> None:
    # 1 of 2 relevant docs in top-3.
    ranking = ["doc-1", "doc-x", "doc-y", "doc-3"]
    relevant = ["doc-1", "doc-3"]
    assert recall_at_k(ranking, relevant, 3) == 0.5


def test_recall_zero_when_no_overlap() -> None:
    assert recall_at_k(["a", "b"], ["c", "d"], 5) == 0.0


def test_recall_empty_relevant_returns_zero() -> None:
    assert recall_at_k(["a", "b"], [], 5) == 0.0


def test_recall_k_zero_returns_zero() -> None:
    assert recall_at_k(["a", "b"], ["a"], 0) == 0.0


# --------------------------------------------------------------------------- #
# mrr@k                                                                       #
# --------------------------------------------------------------------------- #
def test_mrr_first_hit_at_rank_1() -> None:
    assert mrr_at_k(["a", "b", "c"], ["a"], 10) == 1.0


def test_mrr_first_hit_at_rank_3() -> None:
    assert mrr_at_k(["x", "y", "a", "b"], ["a"], 10) == pytest.approx(1.0 / 3.0)


def test_mrr_no_hit_in_topk_is_zero() -> None:
    assert mrr_at_k(["x", "y", "z"], ["a"], 3) == 0.0


def test_mrr_empty_relevant_is_zero() -> None:
    assert mrr_at_k(["a", "b"], [], 5) == 0.0


# --------------------------------------------------------------------------- #
# nDCG@k                                                                      #
# --------------------------------------------------------------------------- #
def test_ndcg_perfect_when_relevant_at_top() -> None:
    # 2 relevant docs at ranks 1+2 → DCG = IDCG → 1.0.
    ranking = ["a", "b", "c", "d"]
    relevant = ["a", "b"]
    assert ndcg_at_k(ranking, relevant, 10) == pytest.approx(1.0)


def test_ndcg_known_value_for_relevant_at_rank_2() -> None:
    # 1 relevant doc at rank 2: DCG = 1/log2(3), IDCG = 1/log2(2) = 1.
    # nDCG = 1/log2(3) ≈ 0.6309.
    ranking = ["x", "a", "y", "z"]
    relevant = ["a"]
    assert ndcg_at_k(ranking, relevant, 10) == pytest.approx(1.0 / math.log2(3))


def test_ndcg_zero_when_no_relevant_in_topk() -> None:
    assert ndcg_at_k(["x", "y", "z"], ["a"], 3) == 0.0


def test_ndcg_empty_relevant_is_zero() -> None:
    assert ndcg_at_k(["a"], [], 5) == 0.0


# --------------------------------------------------------------------------- #
# Registry                                                                    #
# --------------------------------------------------------------------------- #
def test_metrics_registry_has_all_three() -> None:
    assert set(METRICS.keys()) == {"recall_at_5", "mrr_at_10", "ndcg_at_10"}
    for k, fn in METRICS.values():
        # Each scorer is callable with the documented signature.
        v = fn(["a", "b"], ["a"], k)
        assert 0.0 <= v <= 1.0
