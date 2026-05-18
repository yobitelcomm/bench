"""Tests for the deterministic scoring strategies used by vision-understanding."""

from __future__ import annotations

from inferencebench_vision.scoring import (
    SCORERS,
    ScoreContext,
    exact_match,
    substring_match,
)


def _ctx(hypothesis: str, reference: str) -> ScoreContext:
    return ScoreContext(reference=reference, hypothesis=hypothesis)


# --------------------------------------------------------------------------- #
# exact_match                                                                 #
# --------------------------------------------------------------------------- #
def test_exact_match_identical() -> None:
    assert exact_match(_ctx("42", "42")) == 1.0


def test_exact_match_case_insensitive() -> None:
    assert exact_match(_ctx("APRIL 17", "april 17")) == 1.0


def test_exact_match_strips_whitespace() -> None:
    assert exact_match(_ctx("  42  \n", "42")) == 1.0


def test_exact_match_rejects_substring() -> None:
    assert exact_match(_ctx("The tallest bar is 42.", "42")) == 0.0


# --------------------------------------------------------------------------- #
# substring_match                                                             #
# --------------------------------------------------------------------------- #
def test_substring_match_hit() -> None:
    assert (
        substring_match(_ctx("The text reads APRIL 17 across the top.", "april 17"))
        == 1.0
    )


def test_substring_match_case_insensitive() -> None:
    assert substring_match(_ctx("invoice 4421 is shown.", "INVOICE 4421")) == 1.0


def test_substring_match_miss() -> None:
    assert substring_match(_ctx("The image shows nothing legible.", "april 17")) == 0.0


# --------------------------------------------------------------------------- #
# Registry                                                                    #
# --------------------------------------------------------------------------- #
def test_scorers_registry_has_all_three() -> None:
    assert set(SCORERS.keys()) == {"exact_match", "substring_match", "judge_llm"}
    for name in ("exact_match", "substring_match"):
        fn = SCORERS[name]
        v = fn(_ctx("a", "a"))
        assert 0.0 <= v <= 1.0
