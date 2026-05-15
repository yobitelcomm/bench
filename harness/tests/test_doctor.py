"""Tests for the doctor diagnostic.

Real NVML interactions are covered by `@pytest.mark.gpu` integration tests on
TestBM. Here we test the framework (CheckResult / DiagnosticReport semantics,
strict mode behavior) without touching real hardware.
"""

from __future__ import annotations

import pytest

from inferencebench.harness.doctor import (
    CheckResult,
    CheckStatus,
    DiagnosticReport,
    run_diagnostic,
)


# --------------------------------------------------------------------------- #
# DiagnosticReport semantics                                                  #
# --------------------------------------------------------------------------- #
def test_empty_report_is_ok() -> None:
    report = DiagnosticReport(checks=[], strict=False)
    assert report.ok is True
    assert report.fail_count == 0
    assert report.warn_count == 0


def test_all_pass_is_ok() -> None:
    report = DiagnosticReport(
        checks=[
            CheckResult(name="gpu0.thermal", status=CheckStatus.PASS, detail="60°C"),
            CheckResult(name="gpu0.ecc", status=CheckStatus.PASS, detail="0 errors"),
        ],
        strict=False,
    )
    assert report.ok


def test_fail_blocks_in_both_modes() -> None:
    checks = [
        CheckResult(name="gpu0.thermal", status=CheckStatus.PASS, detail=""),
        CheckResult(name="gpu0.ecc", status=CheckStatus.FAIL, detail="1 uncorrected"),
    ]
    assert DiagnosticReport(checks=checks, strict=False).ok is False
    assert DiagnosticReport(checks=checks, strict=True).ok is False


def test_warn_only_blocks_in_strict_mode() -> None:
    checks = [
        CheckResult(name="gpu0.thermal", status=CheckStatus.PASS, detail=""),
        CheckResult(name="gpu0.ecc", status=CheckStatus.WARN, detail="ECC disabled"),
    ]
    assert DiagnosticReport(checks=checks, strict=False).ok is True
    assert DiagnosticReport(checks=checks, strict=True).ok is False


def test_skip_never_blocks() -> None:
    checks = [
        CheckResult(name="gpu0.thermal", status=CheckStatus.SKIP, detail="no driver"),
        CheckResult(name="gpu0.ecc", status=CheckStatus.SKIP, detail="no driver"),
    ]
    assert DiagnosticReport(checks=checks, strict=False).ok is True
    assert DiagnosticReport(checks=checks, strict=True).ok is True


def test_fail_count_and_warn_count() -> None:
    checks = [
        CheckResult(name="a", status=CheckStatus.PASS, detail=""),
        CheckResult(name="b", status=CheckStatus.WARN, detail=""),
        CheckResult(name="c", status=CheckStatus.WARN, detail=""),
        CheckResult(name="d", status=CheckStatus.FAIL, detail=""),
    ]
    report = DiagnosticReport(checks=checks, strict=False)
    assert report.fail_count == 1
    assert report.warn_count == 2


# --------------------------------------------------------------------------- #
# run_diagnostic on a host without NVIDIA                                     #
# --------------------------------------------------------------------------- #
def test_run_diagnostic_skips_gracefully_without_nvidia() -> None:
    """On any host that lacks pynvml/driver, run_diagnostic returns SKIP, never raises."""
    report = run_diagnostic()
    # Either skipped entirely (no NVML) or all checks ran successfully
    assert isinstance(report, DiagnosticReport)
    assert report.checks  # at least one entry (the nvml-skip line)


# --------------------------------------------------------------------------- #
# GPU integration (TestBM only)                                               #
# --------------------------------------------------------------------------- #
@pytest.mark.gpu
def test_run_diagnostic_on_nvidia_host() -> None:
    """On a host with NVIDIA + healthy GPU(s), all checks should PASS or WARN, no FAIL."""
    report = run_diagnostic(strict=False)
    assert report.ok, f"expected OK, got: {[(c.name, c.status, c.detail) for c in report.checks]}"
    # Check there's at least one per-gpu check
    gpu_checks = [c for c in report.checks if c.name.startswith("gpu")]
    assert len(gpu_checks) >= 4, (
        f"expected ≥4 GPU checks (thermal, ECC, memory, throttle), got {len(gpu_checks)}"
    )
