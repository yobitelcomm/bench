"""Tests for the deterministic MT scoring strategies."""

from __future__ import annotations

import pytest

from inferencebench_mt.scoring import SCORERS, bleu_token, chrf, exact_match


# --------------------------------------------------------------------------- #
# chrf                                                                        #
# --------------------------------------------------------------------------- #
def test_chrf_identical_strings_score_one() -> None:
    assert chrf("Bonjour le monde", "Bonjour le monde") == 1.0


def test_chrf_completely_different_scores_low() -> None:
    # Two strings with zero character n-gram overlap.
    score = chrf("aaaa", "bbbb")
    assert score == 0.0


def test_chrf_hand_computed_oracle_abc_abd() -> None:
    """Hand-computed oracle for chrF.

    reference = "abc", hypothesis = "abd", n=2, beta=2.
    1-grams: ref={a,b,c} hyp={a,b,d}, overlap=2, hyp_total=3, ref_total=3
    2-grams: ref={ab,bc} hyp={ab,bd}, overlap=1, hyp_total=2, ref_total=2
    aggregate: match=3 hyp=5 ref=5 → p=r=0.6
    F_2 = (1+4)*p*r / (4*p + r) = 5*0.36 / 3.0 = 0.6
    """
    assert chrf("abc", "abd", n=2, beta=2.0) == pytest.approx(0.6)


def test_chrf_partial_overlap_in_zero_one_band() -> None:
    score = chrf("Bonjour le monde", "Bonsoir le monde")
    assert 0.0 < score < 1.0


def test_chrf_empty_strings_both_empty_score_one() -> None:
    assert chrf("", "") == 1.0


def test_chrf_only_one_side_empty_scores_zero() -> None:
    assert chrf("hello", "") == 0.0
    assert chrf("", "hello") == 0.0


def test_chrf_whitespace_normalisation() -> None:
    # Multiple spaces collapse — same chrF as single-space version.
    assert chrf("a b c", "a  b   c") == 1.0


def test_chrf_n_must_be_positive() -> None:
    with pytest.raises(ValueError, match="n >= 1"):
        chrf("a", "a", n=0)


# --------------------------------------------------------------------------- #
# bleu_token                                                                  #
# --------------------------------------------------------------------------- #
def test_bleu_token_identical_scores_one() -> None:
    # Long enough that 4-grams exist on both sides.
    s = "the quick brown fox jumps over the lazy dog"
    assert bleu_token(s, s) == pytest.approx(1.0)


def test_bleu_token_completely_disjoint_scores_zero() -> None:
    assert bleu_token("alpha beta gamma delta epsilon", "one two three four five") == 0.0


def test_bleu_token_brevity_penalty_applies_for_short_hypothesis() -> None:
    """Hypothesis shorter than reference → brevity penalty < 1 → score < 1."""
    ref = "the cat sat on the mat"
    hyp = "the cat sat on the"  # missing trailing token; all 4-grams present
    score = bleu_token(ref, hyp)
    assert 0.0 < score < 1.0


def test_bleu_token_empty_returns_zero() -> None:
    assert bleu_token("", "anything goes here") == 0.0
    assert bleu_token("anything goes here", "") == 0.0


def test_bleu_token_too_short_for_ngrams_returns_zero() -> None:
    # 3 tokens — no 4-grams possible.
    assert bleu_token("a b c", "a b c") == 0.0


def test_bleu_token_invalid_max_n() -> None:
    with pytest.raises(ValueError, match="max_n >= 1"):
        bleu_token("a b c d", "a b c d", max_n=0)


# --------------------------------------------------------------------------- #
# exact_match                                                                 #
# --------------------------------------------------------------------------- #
def test_exact_match_identical() -> None:
    assert exact_match("Paris", "Paris") == 1.0


def test_exact_match_case_insensitive_and_strip() -> None:
    assert exact_match("  Bonjour  ", "BONJOUR") == 1.0


def test_exact_match_substring_not_a_match() -> None:
    assert exact_match("Paris", "The capital is Paris") == 0.0


# --------------------------------------------------------------------------- #
# Registry                                                                    #
# --------------------------------------------------------------------------- #
def test_scorers_registry_has_all_three() -> None:
    assert set(SCORERS.keys()) == {"chrf", "bleu_token", "exact_match"}
    for name, fn in SCORERS.items():
        v = fn("hello world", "hello world")
        assert 0.0 <= v <= 1.0, f"{name} returned out-of-band value {v}"
