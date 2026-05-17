"""Subprocess-based unit-test runner for the code-generation plugin.

Executes a model-generated Python solution alongside fixture unit tests in
an isolated ``python -I`` subprocess with a wall-clock timeout. **This is
not a real sandbox.** See ``README.md`` for the safety boundary; the
shortlist of forbidden-import heuristics here is defence-in-depth, not
defence-in-full.

Each :func:`run_unit_tests` invocation writes solution + tests to a
temporary file, runs it under ``subprocess.run`` with the supplied
``timeout_s``, captures stdout/stderr, and returns a :class:`RunResult`.
The temp file is always deleted via ``try/finally`` even on timeout or
unhandled errors.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass

# Substrings that immediately disqualify a solution. Cheap pre-scan, not
# a parser — there are dozens of ways around it (eval, __import__, etc).
# That's fine; this is one of several layers, and the bundled fixtures
# never exercise stdlib edges.
_FORBIDDEN_IMPORTS: tuple[str, ...] = (
    "subprocess",
    "os.system",
    "socket",
    "urllib",
    "multiprocessing",
    "ctypes",
)


@dataclass(frozen=True, slots=True)
class RunResult:
    """Outcome of one subprocess execution.

    ``passed`` is True only when the subprocess exited 0 within the wall
    clock. ``timeout`` is True when ``subprocess.TimeoutExpired`` fired
    (in which case ``duration_s`` reflects the timeout, not the real
    wall time). ``stdout`` and ``stderr`` are decoded UTF-8 strings.
    """

    passed: bool
    stdout: str
    stderr: str
    timeout: bool
    duration_s: float


def _scan_forbidden(solution: str) -> str | None:
    """Return the first forbidden token found in ``solution`` or None.

    Case-sensitive substring match; we accept the false-negative risk
    in exchange for a vanishingly small false-positive rate.
    """
    for token in _FORBIDDEN_IMPORTS:
        if token in solution:
            return token
    return None


def run_unit_tests(
    solution: str,
    tests: str,
    *,
    timeout_s: float = 5.0,
) -> RunResult:
    """Execute ``solution + tests`` in an isolated subprocess and report pass/fail.

    Arguments:
        solution: Python code defining the function under test.
        tests: Python code that imports / references the function (already
            in scope because solution + tests are concatenated into one
            file) and exercises it with ``assert`` statements.
        timeout_s: Wall-clock budget. The subprocess is killed when it
            elapses and :class:`RunResult` is returned with ``timeout=True``.

    Returns:
        :class:`RunResult` with ``passed``, captured streams, timeout flag,
        and observed duration in seconds.
    """
    forbidden = _scan_forbidden(solution)
    if forbidden is not None:
        return RunResult(
            passed=False,
            stdout="",
            stderr=f"forbidden_import: solution references {forbidden!r}",
            timeout=False,
            duration_s=0.0,
        )

    body = solution + "\n\n# --- tests ---\n" + tests + "\n"
    # delete=False so we control the lifetime; the finally block removes it.
    tmp = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".py",
        delete=False,
        encoding="utf-8",
    )
    tmp_path = tmp.name
    try:
        tmp.write(body)
        tmp.close()
        start = time.perf_counter()
        try:
            completed = subprocess.run(
                [sys.executable, "-I", tmp_path],
                capture_output=True,
                timeout=timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            duration = time.perf_counter() - start
            stdout_bytes = exc.stdout or b""
            stderr_bytes = exc.stderr or b""
            return RunResult(
                passed=False,
                stdout=stdout_bytes.decode("utf-8", errors="replace"),
                stderr=stderr_bytes.decode("utf-8", errors="replace"),
                timeout=True,
                duration_s=duration,
            )
        duration = time.perf_counter() - start
        stdout = completed.stdout.decode("utf-8", errors="replace")
        stderr = completed.stderr.decode("utf-8", errors="replace")
        return RunResult(
            passed=completed.returncode == 0,
            stdout=stdout,
            stderr=stderr,
            timeout=False,
            duration_s=duration,
        )
    finally:
        # Always remove the temp file — even on TimeoutExpired / KeyboardInterrupt.
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
