"""Tests for the three deterministic scoring strategies (WER / CER / EM)."""

from __future__ import annotations

import pytest

from inferencebench_voice.scoring import (
    SCORERS,
    _levenshtein,
    cer,
    exact_match,
    wer,
)


# --------------------------------------------------------------------------- #
# Levenshtein primitive                                                       #
# --------------------------------------------------------------------------- #
def test_levenshtein_empty_inputs() -> None:
    assert _levenshtein([], []) == 0
    assert _levenshtein(["a"], []) == 1
    assert _levenshtein([], ["a", "b"]) == 2


def test_levenshtein_single_substitution() -> None:
    # "kitten" -> "sitten" is one substitution.
    assert _levenshtein(list("kitten"), list("sitten")) == 1


def test_levenshtein_classic_kitten_sitting() -> None:
    # Distance is 3: kitten -> sitten -> sittin -> sitting
    assert _levenshtein(list("kitten"), list("sitting")) == 3


# --------------------------------------------------------------------------- #
# WER                                                                         #
# --------------------------------------------------------------------------- #
def test_wer_perfect_match() -> None:
    assert wer("hello world", "hello world") == 0.0


def test_wer_case_insensitive() -> None:
    assert wer("Hello World", "hello world") == 0.0


def test_wer_one_word_substitution_in_three() -> None:
    # 1 word changed out of 3 reference words: 1/3.
    assert wer("the cat sat", "the dog sat") == pytest.approx(1.0 / 3.0)


def test_wer_stub_corruption_matches_one_in_nine() -> None:
    # Models the plugin's _synthesise_hypothesis: drop last word + "end".
    # Reference has 9 words; the stub substitutes 1 word -> WER = 1/9.
    ref = "the quick brown fox jumps over the lazy dog"
    hyp = "the quick brown fox jumps over the lazy end"
    assert wer(ref, hyp) == pytest.approx(1.0 / 9.0)


def test_wer_total_mismatch_caps_at_one() -> None:
    assert wer("alpha beta", "gamma delta") == 1.0


def test_wer_empty_reference_handles_gracefully() -> None:
    assert wer("", "") == 0.0
    assert wer("", "anything") == 1.0


# --------------------------------------------------------------------------- #
# CER                                                                         #
# --------------------------------------------------------------------------- #
def test_cer_perfect_match() -> None:
    assert cer("hello", "hello") == 0.0


def test_cer_one_char_swap() -> None:
    # "hello" vs "hallo": 1 substitution out of 5 chars -> 0.2
    assert cer("hello", "hallo") == pytest.approx(0.2)


def test_cer_caps_at_one() -> None:
    assert cer("a", "completely different") == 1.0


# --------------------------------------------------------------------------- #
# exact_match (error rate flavour for this plugin)                            #
# --------------------------------------------------------------------------- #
def test_exact_match_zero_when_equal() -> None:
    assert exact_match("Paris", "paris") == 0.0


def test_exact_match_one_when_different() -> None:
    assert exact_match("Paris", "Lyon") == 1.0


# --------------------------------------------------------------------------- #
# Registry                                                                    #
# --------------------------------------------------------------------------- #
def test_scorers_registry_has_all_three() -> None:
    assert set(SCORERS.keys()) == {"wer", "cer", "exact_match"}
    for fn in SCORERS.values():
        v = fn("a b", "a b")
        assert 0.0 <= v <= 1.0
