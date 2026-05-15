"""Tests for percentile + bootstrap-CI math."""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from inferencebench.harness.metrics import (
    Percentiles,
    bootstrap_percentile_ci,
)


# --------------------------------------------------------------------------- #
# Basic Percentiles behaviour                                                 #
# --------------------------------------------------------------------------- #
def test_percentiles_empty_input_raises() -> None:
    with pytest.raises(ValueError, match="at least one finite"):
        Percentiles([])


def test_percentiles_drops_non_finite_values() -> None:
    """NaN / inf samples are silently dropped."""
    p = Percentiles([1.0, 2.0, 3.0, float("nan"), float("inf"), -float("inf"), 4.0])
    assert p.n == 4
    assert p.min == 1.0
    assert p.max == 4.0


def test_percentiles_basic_values() -> None:
    samples = list(range(1, 101))  # 1..100
    p = Percentiles(samples, bootstrap=False)
    # numpy linear method gives p50 = (sample[49]+sample[50])/2 = 50.5
    assert p.p50 == pytest.approx(50.5)
    # p99 is at the 0.99 quantile (between 99 and 100)
    assert p.p99 == pytest.approx(99.01, abs=0.1)
    assert p.p99_9 == pytest.approx(99.901, abs=0.1)


def test_percentiles_with_bootstrap_attaches_cis() -> None:
    rng = np.random.default_rng(0)
    samples = rng.normal(loc=100, scale=10, size=500)
    p = Percentiles(samples, bootstrap=True, n_resamples=500, seed=1)
    assert len(p.cis) == len(p.percentiles)
    for ci in p.cis:
        assert ci.ci_low <= ci.estimate <= ci.ci_high


def test_percentiles_attribute_access_for_ci() -> None:
    p = Percentiles([1, 2, 3, 4, 5], bootstrap=True, n_resamples=200, seed=42)
    assert math.isfinite(p.p50_ci_low)
    assert math.isfinite(p.p50_ci_high)
    assert p.p50_ci_low <= p.p50 <= p.p50_ci_high


def test_percentiles_unknown_attribute_raises() -> None:
    p = Percentiles([1, 2, 3])
    with pytest.raises(AttributeError):
        p.no_such_thing  # type: ignore[attr-defined]


def test_percentiles_as_dict_has_expected_keys() -> None:
    p = Percentiles([1, 2, 3, 4, 5], bootstrap=True, n_resamples=200)
    d = p.as_dict()
    for k in ("n", "mean", "min", "max", "p50", "p99", "p99_ci_low", "p99_ci_high", "p99_9"):
        assert k in d, f"missing {k}"


# --------------------------------------------------------------------------- #
# bootstrap_percentile_ci direct                                              #
# --------------------------------------------------------------------------- #
def test_bootstrap_ci_degenerate_single_sample() -> None:
    """One sample → CI collapses to that value."""
    arr = np.array([42.0])
    ci = bootstrap_percentile_ci(arr, 50)
    assert ci.ci_low == 42.0 == ci.ci_high == ci.estimate


def test_bootstrap_ci_normal_distribution_known_p50() -> None:
    """For a large normal sample, P50 ≈ mean, CI tight."""
    rng = np.random.default_rng(123)
    arr = rng.normal(loc=100.0, scale=5.0, size=2000)
    ci = bootstrap_percentile_ci(arr, 50, n_resamples=500, rng=rng)
    assert ci.estimate == pytest.approx(100.0, abs=1.0)
    assert ci.ci_high - ci.ci_low < 3.0  # tight CI for N=2000


def test_bootstrap_ci_widens_for_p99_vs_p50() -> None:
    """P99 has more uncertainty than P50 (tail estimation)."""
    rng = np.random.default_rng(7)
    arr = rng.normal(loc=0, scale=1, size=200)
    ci50 = bootstrap_percentile_ci(arr, 50, n_resamples=500, rng=rng)
    ci99 = bootstrap_percentile_ci(arr, 99, n_resamples=500, rng=rng)
    assert (ci99.ci_high - ci99.ci_low) > (ci50.ci_high - ci50.ci_low)


# --------------------------------------------------------------------------- #
# Property tests                                                              #
# --------------------------------------------------------------------------- #
@given(
    st.lists(
        st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False),
        min_size=2,
        max_size=200,
    )
)
@settings(max_examples=30, deadline=None)
def test_percentile_monotonic_in_p(values: list[float]) -> None:
    """P_k is non-decreasing in k."""
    p = Percentiles(values, bootstrap=False)
    sorted_p = sorted(p.percentiles)
    prev = -float("inf")
    for q in sorted_p:
        v = p._values[q]
        assert v >= prev - 1e-9, f"P{q} = {v} < prev = {prev}"
        prev = v
