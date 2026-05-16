"""Tests for the power/energy summariser."""

from __future__ import annotations

import math

from inferencebench.harness.drivers import Sample
from inferencebench.harness.metrics import EnergyReport, summarise_energy
from inferencebench.harness.telemetry import GPUSample, RAPLSample


# --------------------------------------------------------------------------- #
# Empty inputs                                                                #
# --------------------------------------------------------------------------- #
def test_empty_inputs_yield_zero_energy() -> None:
    report = summarise_energy([], [], [], duration_s=10.0)
    assert isinstance(report, EnergyReport)
    assert report.gpu_energy_joules == 0.0
    assert report.rapl_energy_joules == 0.0
    assert math.isnan(report.joules_per_token)
    assert math.isnan(report.joules_per_request)


# --------------------------------------------------------------------------- #
# GPU aggregation                                                             #
# --------------------------------------------------------------------------- #
def _gpu_sample(t_ms: float, power_w: float) -> GPUSample:
    return GPUSample(t_ms=t_ms, devices=({"power_w": power_w},))


def test_gpu_constant_power_integrates_to_energy() -> None:
    """100 W for 10 s ≈ 1000 J."""
    series = [_gpu_sample(t * 1000.0, 100.0) for t in range(11)]  # 0..10 s
    report = summarise_energy(series, [], [], duration_s=10.0)
    assert report.gpu_power_avg_w == 100.0
    assert report.gpu_power_peak_w == 100.0
    assert report.gpu_energy_joules == 1000.0  # 100 W * 10 s


def test_gpu_peak_distinct_from_avg() -> None:
    series = [
        _gpu_sample(0, 50.0),
        _gpu_sample(1000, 200.0),
        _gpu_sample(2000, 50.0),
    ]
    report = summarise_energy(series, [], [], duration_s=2.0)
    assert report.gpu_power_peak_w == 200.0
    assert 50 < report.gpu_power_avg_w < 200


def test_gpu_handles_missing_power_field() -> None:
    """If a sample dict has no power_w, treat as 0 W for that sample."""
    series = [
        GPUSample(t_ms=0, devices=({"power_w": 100.0},)),
        GPUSample(t_ms=1000, devices=({"util_gpu_pct": 50},)),  # no power
    ]
    report = summarise_energy(series, [], [], duration_s=1.0)
    # Trapezoidal of (100, 0) over 1s = 50 J
    assert report.gpu_energy_joules == 50.0


# --------------------------------------------------------------------------- #
# RAPL aggregation                                                            #
# --------------------------------------------------------------------------- #
def _rapl_sample(t_ms: float, energy_uj: int) -> RAPLSample:
    return RAPLSample(t_ms=t_ms, domains=({"name": "package-0", "energy_uj": energy_uj},))


def test_rapl_integrates_delta() -> None:
    """First 100M uJ, last 1100M uJ → delta = 1000M uJ = 1000 J."""
    series = [_rapl_sample(0, 100_000_000), _rapl_sample(10_000, 1_100_000_000)]
    report = summarise_energy([], series, [], duration_s=10.0)
    assert report.rapl_energy_joules == 1000.0


def test_rapl_wraparound_is_treated_as_zero() -> None:
    """If last < first (counter wrap), skip that domain instead of negative energy."""
    series = [_rapl_sample(0, 5_000_000_000), _rapl_sample(1000, 1_000_000_000)]
    report = summarise_energy([], series, [], duration_s=1.0)
    assert report.rapl_energy_joules == 0.0


# --------------------------------------------------------------------------- #
# Joules-per-token / joules-per-request                                       #
# --------------------------------------------------------------------------- #
def test_joules_per_token_basic() -> None:
    """1000 J total, 1000 output tokens → 1 J/token."""
    gpu_series = [_gpu_sample(t * 1000.0, 100.0) for t in range(11)]
    samples = [
        Sample(
            request_idx=i,
            arrival_ms=0,
            start_ms=0,
            ttft_ms=10,
            total_ms=100,
            tpot_ms=1,
            tokens_in=10,
            tokens_out=100,
            cost_usd=0,
            finish_reason="stop",
            ok=True,
        )
        for i in range(10)  # 10 requests * 100 tokens = 1000 tokens
    ]
    report = summarise_energy(gpu_series, [], samples, duration_s=10.0)
    assert report.joules_per_token == 1.0
    assert report.joules_per_request == 100.0


def test_failed_requests_excluded_from_token_count() -> None:
    """ok=False samples don't contribute tokens or request count."""
    gpu_series = [_gpu_sample(0, 100.0), _gpu_sample(1000, 100.0)]
    samples = [
        Sample(
            request_idx=0,
            arrival_ms=0,
            start_ms=0,
            ttft_ms=10,
            total_ms=100,
            tpot_ms=1,
            tokens_in=4,
            tokens_out=10,
            cost_usd=0,
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
            cost_usd=0,
            finish_reason="error",
            ok=False,
        ),
    ]
    report = summarise_energy(gpu_series, [], samples, duration_s=1.0)
    assert report.joules_per_token == 100.0 / 10.0
    assert report.joules_per_request == 100.0  # only 1 ok request


def test_zero_tokens_yields_nan_jpt() -> None:
    gpu_series = [_gpu_sample(0, 100.0), _gpu_sample(1000, 100.0)]
    report = summarise_energy(gpu_series, [], [], duration_s=1.0)
    assert math.isnan(report.joules_per_token)
    assert math.isnan(report.joules_per_request)
