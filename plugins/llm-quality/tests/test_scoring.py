"""Tests for the deterministic scoring strategies."""

from __future__ import annotations

import pytest

from inferencebench_quality.scoring import (
    SCORERS,
    ScoreContext,
    exact_match,
    f1_token,
    substring_match,
)


def _ctx(hypothesis: str, reference: str) -> ScoreContext:
    return ScoreContext(reference=reference, hypothesis=hypothesis)


# --------------------------------------------------------------------------- #
# exact_match                                                                 #
# --------------------------------------------------------------------------- #
def test_exact_match_identical() -> None:
    assert exact_match(_ctx("Paris", "Paris")) == 1.0


def test_exact_match_case_insensitive() -> None:
    assert exact_match(_ctx("PARIS", "paris")) == 1.0


def test_exact_match_strips_whitespace() -> None:
    assert exact_match(_ctx("  8  \n", "8")) == 1.0


def test_exact_match_rejects_substring() -> None:
    assert exact_match(_ctx("The answer is 8.", "8")) == 0.0


def test_exact_match_empty_strings_match() -> None:
    assert exact_match(_ctx("", "")) == 1.0


# --------------------------------------------------------------------------- #
# substring_match                                                             #
# --------------------------------------------------------------------------- #
def test_substring_match_hit() -> None:
    assert substring_match(_ctx("The capital of France is Paris.", "Paris")) == 1.0


def test_substring_match_case_insensitive() -> None:
    assert substring_match(_ctx("paris is lovely", "Paris")) == 1.0


def test_substring_match_miss() -> None:
    assert substring_match(_ctx("The capital of France is Lyon.", "Paris")) == 0.0


def test_substring_match_empty_reference_is_vacuous_hit() -> None:
    assert substring_match(_ctx("anything", "")) == 1.0


# --------------------------------------------------------------------------- #
# f1_token                                                                    #
# --------------------------------------------------------------------------- #
def test_f1_token_exact_overlap() -> None:
    assert f1_token(_ctx("the cat sat", "the cat sat")) == 1.0


def test_f1_token_partial_overlap() -> None:
    assert f1_token(_ctx("the cat ran fast", "the cat sat down")) == pytest.approx(0.5)


def test_f1_token_zero_when_either_side_empty() -> None:
    assert f1_token(_ctx("", "anything")) == 0.0
    assert f1_token(_ctx("anything", "")) == 0.0


def test_f1_token_case_insensitive() -> None:
    assert f1_token(_ctx("THE Cat SAT", "the cat sat")) == 1.0


def test_f1_token_no_overlap_is_zero() -> None:
    assert f1_token(_ctx("alpha beta", "gamma delta")) == 0.0


# --------------------------------------------------------------------------- #
# Registry                                                                    #
# --------------------------------------------------------------------------- #
def test_scorers_registry_has_all_four() -> None:
    assert set(SCORERS.keys()) == {
        "exact_match",
        "substring_match",
        "f1_token",
        "judge_llm",
    }
    # Each deterministic scorer is callable with a ScoreContext and returns a
    # float in [0, 1]. We don't call the judge scorer here — that needs a
    # mocked ModelClient (covered in test_judge_scoring.py).
    for name in ("exact_match", "substring_match", "f1_token"):
        fn = SCORERS[name]
        v = fn(_ctx("a", "a"))
        assert 0.0 <= v <= 1.0
