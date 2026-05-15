"""Tests for the warmup + convergence gate."""

from __future__ import annotations

import pytest

from inferencebench.harness.convergence import ConvergenceGate


# --------------------------------------------------------------------------- #
# Validation                                                                  #
# --------------------------------------------------------------------------- #
def test_gate_validates_inputs() -> None:
    with pytest.raises(ValueError, match="warmup_runs"):
        ConvergenceGate(warmup_runs=-1)
    with pytest.raises(ValueError, match="window"):
        ConvergenceGate(window=1)
    with pytest.raises(ValueError, match="threshold"):
        ConvergenceGate(threshold=0)


# --------------------------------------------------------------------------- #
# Warmup behaviour                                                            #
# --------------------------------------------------------------------------- #
def test_warmup_samples_are_ignored() -> None:
    gate = ConvergenceGate(warmup_runs=3, window=3, threshold=0.05)
    # Three warmup samples — should not even enter the buffer
    for v in (1.0, 2.0, 1.5):
        gate.update(v)
    assert gate.state.n_warmed == 0
    assert not gate.is_converged


# --------------------------------------------------------------------------- #
# Convergence                                                                 #
# --------------------------------------------------------------------------- #
def test_constant_stream_converges_quickly() -> None:
    gate = ConvergenceGate(warmup_runs=0, window=5, threshold=0.05)
    for _ in range(5):
        gate.update(100.0)
    assert gate.is_converged
    assert gate.state.cov == pytest.approx(0.0)


def test_noisy_stream_within_threshold_converges() -> None:
    gate = ConvergenceGate(warmup_runs=0, window=30, threshold=0.05)
    # ± 1% noise around mean 100
    import random

    rng = random.Random(42)
    for _ in range(30):
        gate.update(100.0 + rng.uniform(-1.0, 1.0))
    assert gate.is_converged


def test_noisy_stream_above_threshold_does_not_converge() -> None:
    gate = ConvergenceGate(warmup_runs=0, window=30, threshold=0.05)
    import random

    rng = random.Random(42)
    for _ in range(30):
        gate.update(100.0 + rng.uniform(-30.0, 30.0))  # ~30% CoV
    assert not gate.is_converged
    assert gate.state.cov > 0.10


def test_window_not_full_does_not_converge() -> None:
    gate = ConvergenceGate(warmup_runs=0, window=10, threshold=0.05)
    for _ in range(5):
        gate.update(100.0)
    # Only 5 in buffer, window=10
    assert not gate.is_converged


# --------------------------------------------------------------------------- #
# Bail-out                                                                    #
# --------------------------------------------------------------------------- #
def test_bails_out_after_max_wait_requests() -> None:
    gate = ConvergenceGate(warmup_runs=0, window=10, threshold=0.001, max_wait_requests=20)
    import random

    rng = random.Random(0)
    # High-variance: never converge under 0.1% threshold
    for _ in range(50):
        gate.update(100.0 + rng.uniform(-50, 50))
    assert gate.bailed_out
    assert not gate.is_converged


# --------------------------------------------------------------------------- #
# Reset                                                                       #
# --------------------------------------------------------------------------- #
def test_reset_clears_state() -> None:
    gate = ConvergenceGate(warmup_runs=0, window=5, threshold=0.05)
    for _ in range(5):
        gate.update(100.0)
    assert gate.is_converged
    gate.reset()
    assert not gate.is_converged
    assert gate.state.n_seen == 0
    assert gate.state.n_warmed == 0


# --------------------------------------------------------------------------- #
# Non-finite handling                                                         #
# --------------------------------------------------------------------------- #
def test_nan_and_inf_are_skipped() -> None:
    gate = ConvergenceGate(warmup_runs=0, window=3, threshold=0.05)
    gate.update(float("nan"))
    gate.update(100.0)
    gate.update(float("inf"))
    gate.update(100.0)
    gate.update(100.0)
    # 3 valid samples in the window, all 100 → converged
    assert gate.is_converged
