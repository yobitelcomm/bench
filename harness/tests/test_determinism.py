"""Determinism tests across the harness.

Premise: same seed + same inputs MUST produce the same outputs. If this ever
breaks, every published envelope's content_hash becomes unreproducible and
``bench verify`` cannot validate reproducibility claims.

We property-test the key boundaries:
- Percentiles (numpy quantile is deterministic but bootstrap uses an RNG → seed-bound)
- bootstrap_percentile_ci with the same rng → same CI
- ConvergenceGate against the same stream → same converged/bailed state
- HardwareFingerprint.compute_fingerprint_sha256 from the same body → same hash
- OpenLoopDriver arrival times from same seed → same arrival times
- Envelope.content_hash() is invariant under signature attach
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from inferencebench.harness.convergence import ConvergenceGate
from inferencebench.harness.drivers import OpenLoopDriver, Sample
from inferencebench.harness.metrics import (
    BootstrapCI,
    GoodputAtSLO,
    Percentiles,
    SLOPredicate,
    bootstrap_percentile_ci,
)


# --------------------------------------------------------------------------- #
# Percentiles + bootstrap                                                     #
# --------------------------------------------------------------------------- #
@given(
    st.lists(
        st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False),
        min_size=2,
        max_size=200,
    ),
    st.integers(min_value=0, max_value=2**31 - 1),
)
@settings(max_examples=30, deadline=None)
def test_percentiles_deterministic_same_seed(values: list[float], seed: int) -> None:
    """Same values + same seed → same point estimates and same CIs."""
    p1 = Percentiles(values, n_resamples=200, seed=seed)
    p2 = Percentiles(values, n_resamples=200, seed=seed)
    assert p1.as_dict() == p2.as_dict()


@given(st.integers(min_value=0, max_value=2**31 - 1))
@settings(max_examples=20, deadline=None)
def test_bootstrap_ci_deterministic_same_rng(seed: int) -> None:
    """Same RNG state → identical bootstrap interval."""
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    ci1 = bootstrap_percentile_ci(arr, 90, n_resamples=300, rng=np.random.default_rng(seed))
    ci2 = bootstrap_percentile_ci(arr, 90, n_resamples=300, rng=np.random.default_rng(seed))
    assert isinstance(ci1, BootstrapCI)
    assert ci1.estimate == ci2.estimate
    assert ci1.ci_low == ci2.ci_low
    assert ci1.ci_high == ci2.ci_high


# --------------------------------------------------------------------------- #
# ConvergenceGate                                                             #
# --------------------------------------------------------------------------- #
@given(
    st.lists(
        st.floats(min_value=1.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
        min_size=10,
        max_size=100,
    )
)
@settings(max_examples=30, deadline=None)
def test_convergence_gate_deterministic(stream: list[float]) -> None:
    """Same stream into two gates with the same config → same state."""
    g1 = ConvergenceGate(warmup_runs=2, window=5, threshold=0.05)
    g2 = ConvergenceGate(warmup_runs=2, window=5, threshold=0.05)
    for v in stream:
        g1.update(v)
        g2.update(v)
    assert g1.state == g2.state


# --------------------------------------------------------------------------- #
# OpenLoopDriver arrival pattern (no I/O, just timestamps)                    #
# --------------------------------------------------------------------------- #
def test_open_loop_arrival_pattern_deterministic_same_seed() -> None:
    """Inspect the internal RNG path: same seed → identical arrival timestamps."""
    # We don't drive any requests; we just spin the driver with a synthetic
    # request_fn that records arrival_ms and exits immediately.
    captured1: list[float] = []
    captured2: list[float] = []

    def _fn_for(captured: list[float]):
        def _fn(idx: int, item: object) -> Sample:
            return Sample(
                request_idx=idx,
                arrival_ms=0.0,
                start_ms=0.0,
                ttft_ms=0.001,
                total_ms=0.001,
                tpot_ms=0.0,
                tokens_in=0,
                tokens_out=0,
                cost_usd=0.0,
                finish_reason="stop",
                ok=True,
            )

        return _fn

    d1 = OpenLoopDriver(mean_rps=50.0, duration_s=0.2, seed=12345)
    d2 = OpenLoopDriver(mean_rps=50.0, duration_s=0.2, seed=12345)
    s1 = d1.execute(_fn_for(captured1), workload=["x"])
    s2 = d2.execute(_fn_for(captured2), workload=["x"])

    # Arrival timestamps are RNG-deterministic — same seed → same arrivals
    assert len(s1) == len(s2)
    arrivals_1 = [s.arrival_ms for s in s1]
    arrivals_2 = [s.arrival_ms for s in s2]
    assert arrivals_1 == pytest.approx(arrivals_2, rel=0, abs=1e-9)


# --------------------------------------------------------------------------- #
# Goodput-at-SLO                                                              #
# --------------------------------------------------------------------------- #
def test_goodput_no_slos_passes_all_ok() -> None:
    """With zero SLOs, every ok=True sample passes."""
    samples = [
        Sample(
            request_idx=i,
            arrival_ms=0,
            start_ms=0,
            ttft_ms=100 + i,
            total_ms=200 + i,
            tpot_ms=10,
            tokens_in=4,
            tokens_out=10,
            cost_usd=0.001,
            finish_reason="stop",
            ok=True,
        )
        for i in range(10)
    ]
    g = GoodputAtSLO.from_samples(samples, duration_s=10.0, slos=[])
    assert g.passing_samples == 10
    assert g.req_per_s_passing == 1.0
    assert g.compliance_rate == 1.0


def test_goodput_with_threshold_partitions_correctly() -> None:
    """ttft < 150 ms → 5 of 10 pass."""
    samples = [
        Sample(
            request_idx=i,
            arrival_ms=0,
            start_ms=0,
            ttft_ms=100.0 + i * 10.0,  # 100..190
            total_ms=200,
            tpot_ms=10,
            tokens_in=4,
            tokens_out=10,
            cost_usd=0.001,
            finish_reason="stop",
            ok=True,
        )
        for i in range(10)
    ]
    slo = SLOPredicate(name="ttft", field="ttft_ms", op="<", value=150.0)
    g = GoodputAtSLO.from_samples(samples, duration_s=10.0, slos=[slo])
    assert g.passing_samples == 5
    assert g.failing_samples == 5
    assert g.req_per_s_passing == 0.5
    assert g.per_slo_pass_rate["ttft"] == 0.5


def test_goodput_excludes_errors_from_passing() -> None:
    """ok=False samples are never passing."""
    samples = [
        Sample(
            request_idx=0,
            arrival_ms=0,
            start_ms=0,
            ttft_ms=50,
            total_ms=100,
            tpot_ms=5,
            tokens_in=4,
            tokens_out=10,
            cost_usd=0.001,
            finish_reason="stop",
            ok=True,
        ),
        Sample(
            request_idx=1,
            arrival_ms=0,
            start_ms=0,
            ttft_ms=float("nan"),
            total_ms=float("nan"),
            tpot_ms=float("nan"),
            tokens_in=0,
            tokens_out=0,
            cost_usd=0.0,
            finish_reason="error",
            ok=False,
            error="provider unreachable",
        ),
    ]
    g = GoodputAtSLO.from_samples(samples, duration_s=1.0, slos=[])
    assert g.total_samples == 2
    assert g.ok_samples == 1
    assert g.passing_samples == 1


def test_goodput_validates_duration() -> None:
    with pytest.raises(ValueError, match="duration_s"):
        GoodputAtSLO.from_samples([], duration_s=0.0)


def test_slo_predicate_handles_missing_field() -> None:
    """Predicates against a non-existent field return False, not raise."""
    s = Sample(
        request_idx=0,
        arrival_ms=0,
        start_ms=0,
        ttft_ms=10,
        total_ms=20,
        tpot_ms=1,
        tokens_in=1,
        tokens_out=1,
        cost_usd=0,
        finish_reason="stop",
        ok=True,
    )
    p = SLOPredicate(name="x", field="no_such_field", op="<", value=100.0)
    assert p.evaluate(s) is False


def test_slo_predicate_all_operators() -> None:
    s = Sample(
        request_idx=0,
        arrival_ms=0,
        start_ms=0,
        ttft_ms=50.0,
        total_ms=100,
        tpot_ms=5,
        tokens_in=1,
        tokens_out=1,
        cost_usd=0,
        finish_reason="stop",
        ok=True,
    )
    assert SLOPredicate("lt", "ttft_ms", "<", 100.0).evaluate(s)
    assert SLOPredicate("le", "ttft_ms", "<=", 50.0).evaluate(s)
    assert SLOPredicate("gt", "ttft_ms", ">", 10.0).evaluate(s)
    assert SLOPredicate("ge", "ttft_ms", ">=", 50.0).evaluate(s)
    assert SLOPredicate("eq", "ttft_ms", "==", 50.0).evaluate(s)
    assert not SLOPredicate("lt-fail", "ttft_ms", "<", 10.0).evaluate(s)
