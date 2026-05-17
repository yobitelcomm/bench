"""Tests for the pure scoring helpers in :mod:`inferencebench_code.scoring`."""

from __future__ import annotations

import pytest

from inferencebench_code.scoring import compute_pass_at_k, extract_python_code


# --------------------------------------------------------------------------- #
# extract_python_code                                                         #
# --------------------------------------------------------------------------- #
def test_extract_python_code_basic_fence() -> None:
    text = "Here is my answer:\n```python\ndef add(a, b):\n    return a + b\n```\nDone."
    assert extract_python_code(text) == "def add(a, b):\n    return a + b"


def test_extract_python_code_py_tag_alias() -> None:
    text = "```py\nx = 1\n```"
    assert extract_python_code(text) == "x = 1"


def test_extract_python_code_no_fence_falls_back_to_whole_text() -> None:
    text = "  def add(a, b):\n    return a + b  \n"
    assert extract_python_code(text) == "def add(a, b):\n    return a + b"


def test_extract_python_code_multi_fence_returns_first() -> None:
    text = (
        "```python\ndef first():\n    return 1\n```\n"
        "And another:\n"
        "```python\ndef second():\n    return 2\n```\n"
    )
    assert extract_python_code(text) == "def first():\n    return 1"


def test_extract_python_code_bare_fence_fallback() -> None:
    text = "```\nx = 42\n```"
    assert extract_python_code(text) == "x = 42"


def test_extract_python_code_strips_outer_whitespace() -> None:
    text = "   \n\n  some_var = 'hello'  \n  \n"
    assert extract_python_code(text) == "some_var = 'hello'"


# --------------------------------------------------------------------------- #
# compute_pass_at_k                                                           #
# --------------------------------------------------------------------------- #
def test_compute_pass_at_1_collapses_to_mean() -> None:
    # pass@1 with single sample per task is just the mean of the bools.
    assert compute_pass_at_k([True, True, False, True], 1) == pytest.approx(0.75)


def test_compute_pass_at_k_all_pass_returns_one() -> None:
    assert compute_pass_at_k([True, True, True, True], 2) == 1.0


def test_compute_pass_at_k_all_fail_returns_zero() -> None:
    assert compute_pass_at_k([False, False, False, False], 2) == 0.0


def test_compute_pass_at_k_empty_results_returns_zero() -> None:
    assert compute_pass_at_k([], 1) == 0.0
