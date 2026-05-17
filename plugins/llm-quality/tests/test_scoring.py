"""Tests for the three deterministic scoring strategies."""

from __future__ import annotations

import pytest

from inferencebench_quality.scoring import (
    SCORERS,
    exact_match,
    f1_token,
    substring_match,
)


# --------------------------------------------------------------------------- #
# exact_match                                                                 #
# --------------------------------------------------------------------------- #
def test_exact_match_identical() -> None:
    assert exact_match("Paris", "Paris") == 1.0


def test_exact_match_case_insensitive() -> None:
    assert exact_match("PARIS", "paris") == 1.0


def test_exact_match_strips_whitespace() -> None:
    assert exact_match("  8  \n", "8") == 1.0


def test_exact_match_rejects_substring() -> None:
    # exact-match is strict: substring is NOT enough.
    assert exact_match("The answer is 8.", "8") == 0.0


def test_exact_match_empty_strings_match() -> None:
    # Both empty → consistent equality.
    assert exact_match("", "") == 1.0


# --------------------------------------------------------------------------- #
# substring_match                                                             #
# --------------------------------------------------------------------------- #
def test_substring_match_hit() -> None:
    assert substring_match("The capital of France is Paris.", "Paris") == 1.0


def test_substring_match_case_insensitive() -> None:
    assert substring_match("paris is lovely", "Paris") == 1.0


def test_substring_match_miss() -> None:
    assert substring_match("The capital of France is Lyon.", "Paris") == 0.0


def test_substring_match_empty_reference_is_vacuous_hit() -> None:
    # Empty substring trivially appears in any string.
    assert substring_match("anything", "") == 1.0


# --------------------------------------------------------------------------- #
# f1_token                                                                    #
# --------------------------------------------------------------------------- #
def test_f1_token_exact_overlap() -> None:
    assert f1_token("the cat sat", "the cat sat") == 1.0


def test_f1_token_partial_overlap() -> None:
    # pred=4 tokens, ref=4 tokens, overlap = {the, cat} = 2
    # precision = 0.5, recall = 0.5, F1 = 0.5
    assert f1_token("the cat ran fast", "the cat sat down") == pytest.approx(0.5)


def test_f1_token_zero_when_either_side_empty() -> None:
    assert f1_token("", "anything") == 0.0
    assert f1_token("anything", "") == 0.0


def test_f1_token_case_insensitive() -> None:
    assert f1_token("THE Cat SAT", "the cat sat") == 1.0


def test_f1_token_no_overlap_is_zero() -> None:
    assert f1_token("alpha beta", "gamma delta") == 0.0


# --------------------------------------------------------------------------- #
# Registry                                                                    #
# --------------------------------------------------------------------------- #
def test_scorers_registry_has_all_three() -> None:
    assert set(SCORERS.keys()) == {"exact_match", "substring_match", "f1_token"}
    # And each entry is callable with two strings and returns a float in [0, 1].
    for fn in SCORERS.values():
        v = fn("a", "a")
        assert 0.0 <= v <= 1.0
