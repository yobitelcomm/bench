"""Tests for the subprocess-based unit-test runner."""

from __future__ import annotations

import tempfile
from pathlib import Path

from inferencebench_code.runner import RunResult, run_unit_tests


# --------------------------------------------------------------------------- #
# Happy paths                                                                 #
# --------------------------------------------------------------------------- #
def test_run_unit_tests_passing_solution() -> None:
    solution = "def add(a, b):\n    return a + b\n"
    tests = "assert add(1, 2) == 3\nassert add(0, 0) == 0\n"
    result = run_unit_tests(solution, tests, timeout_s=5.0)
    assert isinstance(result, RunResult)
    assert result.passed is True
    assert result.timeout is False
    assert result.duration_s >= 0.0


def test_run_unit_tests_failing_assert_reports_assertionerror() -> None:
    solution = "def add(a, b):\n    return a - b\n"  # intentionally wrong
    tests = "assert add(1, 2) == 3\n"
    result = run_unit_tests(solution, tests, timeout_s=5.0)
    assert result.passed is False
    assert result.timeout is False
    assert "AssertionError" in result.stderr


# --------------------------------------------------------------------------- #
# Timeout                                                                     #
# --------------------------------------------------------------------------- #
def test_run_unit_tests_timeout() -> None:
    solution = "def loop():\n    while True:\n        pass\n"
    tests = "loop()\n"
    result = run_unit_tests(solution, tests, timeout_s=1.0)
    assert result.timeout is True
    assert result.passed is False
    # subprocess.run(timeout=...) returns control near the deadline; allow
    # a small jitter band on the wall clock.
    assert result.duration_s >= 0.9


# --------------------------------------------------------------------------- #
# Forbidden-import pre-scan                                                   #
# --------------------------------------------------------------------------- #
def test_run_unit_tests_refuses_subprocess_import() -> None:
    solution = "import subprocess\n\ndef nope():\n    return 0\n"
    tests = "assert nope() == 0\n"
    result = run_unit_tests(solution, tests, timeout_s=5.0)
    assert result.passed is False
    assert result.timeout is False
    assert "forbidden_import" in result.stderr
    assert "subprocess" in result.stderr


def test_run_unit_tests_refuses_socket_import() -> None:
    solution = "import socket\n\ndef nope():\n    return 0\n"
    tests = "assert nope() == 0\n"
    result = run_unit_tests(solution, tests, timeout_s=5.0)
    assert result.passed is False
    assert "forbidden_import" in result.stderr


# --------------------------------------------------------------------------- #
# Temp file cleanup                                                           #
# --------------------------------------------------------------------------- #
def test_run_unit_tests_cleans_up_temp_files() -> None:
    """5 runs back-to-back must not leave temp files behind in the system tempdir.

    We capture the count of *.py files in the tempdir before/after; the
    delta must be 0 because :func:`run_unit_tests` removes its file in a
    ``finally`` block on every exit path.
    """
    tmp = Path(tempfile.gettempdir())

    def count_py_files() -> int:
        # tmp can contain other test artifacts; only count the
        # NamedTemporaryFile-suffixed ones our runner creates.
        return sum(1 for _ in tmp.glob("tmp*.py"))

    before = count_py_files()
    solution = "def add(a, b):\n    return a + b\n"
    tests = "assert add(1, 2) == 3\n"
    for _ in range(5):
        run_unit_tests(solution, tests, timeout_s=5.0)
    after = count_py_files()
    assert after == before
