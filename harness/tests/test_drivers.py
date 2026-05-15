"""Tests for the request drivers (open-loop + closed-loop).

We use a synthetic request_fn that sleeps for a deterministic duration and
returns a Sample. No real model invocation.
"""

from __future__ import annotations

import time
from typing import Any

import pytest

from inferencebench.harness.drivers import (
    ClosedLoopDriver,
    OpenLoopDriver,
    Sample,
)


def _make_fake_request_fn(latency_ms: float = 20.0):
    """Build a request_fn that sleeps `latency_ms` and returns a valid Sample."""

    def _fn(idx: int, item: Any) -> Sample:
        t0 = time.perf_counter()
        time.sleep(latency_ms / 1000.0)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return Sample(
            request_idx=idx,
            arrival_ms=0.0,  # driver overwrites
            start_ms=0.0,  # driver overwrites
            ttft_ms=elapsed_ms / 2,
            total_ms=elapsed_ms,
            tpot_ms=elapsed_ms / 10,
            tokens_in=4,
            tokens_out=10,
            cost_usd=0.0001,
            finish_reason="stop",
            ok=True,
        )

    return _fn


# --------------------------------------------------------------------------- #
# OpenLoopDriver                                                              #
# --------------------------------------------------------------------------- #
def test_open_loop_validates_inputs() -> None:
    with pytest.raises(ValueError, match="mean_rps"):
        OpenLoopDriver(mean_rps=0, duration_s=1.0)
    with pytest.raises(ValueError, match="duration_s"):
        OpenLoopDriver(mean_rps=1.0, duration_s=0)


def test_open_loop_requires_workload() -> None:
    driver = OpenLoopDriver(mean_rps=5.0, duration_s=0.2)
    with pytest.raises(ValueError, match="workload must contain"):
        driver.execute(_make_fake_request_fn(), workload=[])


def test_open_loop_produces_samples() -> None:
    driver = OpenLoopDriver(mean_rps=20.0, duration_s=0.5, seed=42)
    samples = driver.execute(_make_fake_request_fn(latency_ms=10), workload=["p1", "p2"])
    # 20 rps * 0.5 s ≈ 10 expected samples; allow stochastic variation
    assert 3 <= len(samples) <= 20
    for s in samples:
        assert isinstance(s, Sample)
        assert s.ok
        assert s.tokens_out == 10
        assert s.arrival_ms >= 0
        assert (
            s.start_ms >= s.arrival_ms - 50
        )  # queue delay can be slightly negative due to scheduling


def test_open_loop_deterministic_with_same_seed() -> None:
    """Same seed produces same arrival pattern → same number of samples (modulo stochastic latency)."""
    d1 = OpenLoopDriver(mean_rps=30.0, duration_s=0.3, seed=123)
    d2 = OpenLoopDriver(mean_rps=30.0, duration_s=0.3, seed=123)
    s1 = d1.execute(_make_fake_request_fn(5), workload=["x"])
    s2 = d2.execute(_make_fake_request_fn(5), workload=["x"])
    # Arrival times should be identical (deterministic from seed)
    assert len(s1) == len(s2)
    for a, b in zip(s1, s2, strict=True):
        assert a.arrival_ms == pytest.approx(b.arrival_ms, rel=0, abs=1e-6)


def test_open_loop_handles_request_fn_failure() -> None:
    """Request errors are recorded as ok=False Samples, not propagated."""

    def _flaky(idx: int, item: Any) -> Sample:
        if idx % 2 == 0:
            raise RuntimeError("simulated provider error")
        return Sample(
            request_idx=idx,
            arrival_ms=0,
            start_ms=0,
            ttft_ms=5,
            total_ms=10,
            tpot_ms=0.5,
            tokens_in=1,
            tokens_out=2,
            cost_usd=0,
            finish_reason="stop",
            ok=True,
        )

    driver = OpenLoopDriver(mean_rps=30.0, duration_s=0.3, seed=0)
    samples = driver.execute(_flaky, workload=["x"])
    assert any(s.ok is False for s in samples)
    assert any(s.ok is True for s in samples)
    for failed in (s for s in samples if not s.ok):
        assert "simulated provider error" in failed.error


def test_open_loop_warmup_discard() -> None:
    driver = OpenLoopDriver(mean_rps=40.0, duration_s=0.3, seed=7, warmup_requests=3)
    samples = driver.execute(_make_fake_request_fn(5), workload=["x"])
    # The first 3 samples are dropped — verify the first surviving idx is ≥ 3
    if samples:
        assert min(s.request_idx for s in samples) >= 3


# --------------------------------------------------------------------------- #
# ClosedLoopDriver                                                            #
# --------------------------------------------------------------------------- #
def test_closed_loop_validates_inputs() -> None:
    with pytest.raises(ValueError, match="concurrency"):
        ClosedLoopDriver(concurrency=0, duration_s=1.0)
    with pytest.raises(ValueError, match="duration_s"):
        ClosedLoopDriver(concurrency=1, duration_s=0)


def test_closed_loop_requires_workload() -> None:
    driver = ClosedLoopDriver(concurrency=2, duration_s=0.2)
    with pytest.raises(ValueError, match="workload must contain"):
        driver.execute(_make_fake_request_fn(), workload=[])


def test_closed_loop_produces_samples() -> None:
    driver = ClosedLoopDriver(concurrency=4, duration_s=0.3)
    samples = driver.execute(_make_fake_request_fn(latency_ms=15), workload=["a", "b"])
    # 4 workers * (300ms / 15ms) = 80 expected; allow slack
    assert 10 <= len(samples) <= 100
    for s in samples:
        assert s.ok
        assert s.extra.get("worker_id") is not None
        assert 0 <= s.extra["worker_id"] < 4


def test_closed_loop_concurrency_one_is_sequential() -> None:
    driver = ClosedLoopDriver(concurrency=1, duration_s=0.2)
    samples = driver.execute(_make_fake_request_fn(latency_ms=20), workload=["x"])
    # All samples come from worker 0
    assert all(s.extra.get("worker_id") == 0 for s in samples)
    # request_idx is contiguous
    indices = [s.request_idx for s in samples]
    assert indices == sorted(indices)


def test_closed_loop_max_requests_caps_total() -> None:
    driver = ClosedLoopDriver(concurrency=4, duration_s=2.0, max_requests=5)
    samples = driver.execute(_make_fake_request_fn(latency_ms=5), workload=["x"])
    assert len(samples) <= 5


def test_closed_loop_warmup_discard() -> None:
    driver = ClosedLoopDriver(concurrency=2, duration_s=0.2, warmup_requests=2)
    samples = driver.execute(_make_fake_request_fn(latency_ms=10), workload=["x"])
    if samples:
        # Sample list is sorted by request_idx; first surviving idx >= 2
        assert samples[0].request_idx >= 2
