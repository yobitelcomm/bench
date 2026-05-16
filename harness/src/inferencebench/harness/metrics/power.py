"""Power + energy metrics from telemetry samples.

Turns the raw GPU/RAPL time series into per-run summaries:

- ``power_avg_w`` — mean power draw across the measurement window
- ``power_peak_w`` — peak power
- ``energy_joules_total`` — area under the power curve (∫ p dt)
- ``joules_per_token`` — total energy / total output tokens
- ``joules_per_request`` — total energy / number of completed requests

Phase 1 uses NVML GPU power directly and RAPL energy counters (delta between
first and last sample). Phase 2 adds wall-plug via IPMI/Redfish.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from inferencebench.harness.drivers import Sample
from inferencebench.harness.telemetry import GPUSample, RAPLSample


@dataclass(frozen=True, slots=True)
class EnergyReport:
    """Aggregated power/energy numbers for one run."""

    gpu_power_avg_w: float
    gpu_power_peak_w: float
    gpu_energy_joules: float
    rapl_energy_joules: float
    total_energy_joules: float
    joules_per_token: float  # NaN if 0 output tokens
    joules_per_request: float  # NaN if 0 successful requests
    duration_s: float


def summarise_energy(
    gpu_series: Iterable[GPUSample],
    rapl_series: Iterable[RAPLSample],
    samples: Iterable[Sample],
    duration_s: float,
) -> EnergyReport:
    """Compute :class:`EnergyReport` from telemetry + sample streams.

    Args:
        gpu_series: GPU samples from :class:`NVMLSampler`.
        rapl_series: RAPL samples from :class:`RAPLSampler`.
        samples: Driver samples (per-request).
        duration_s: Measurement window in seconds.
    """
    gpu_list = list(gpu_series)
    rapl_list = list(rapl_series)
    sample_list = list(samples)

    gpu_avg_w, gpu_peak_w, gpu_energy_j = _aggregate_gpu(gpu_list)
    rapl_energy_j = _aggregate_rapl(rapl_list)

    total_energy = gpu_energy_j + rapl_energy_j
    tokens_out = sum(s.tokens_out for s in sample_list if s.ok)
    ok_requests = sum(1 for s in sample_list if s.ok)

    jpt = total_energy / tokens_out if tokens_out else float("nan")
    jpr = total_energy / ok_requests if ok_requests else float("nan")

    return EnergyReport(
        gpu_power_avg_w=gpu_avg_w,
        gpu_power_peak_w=gpu_peak_w,
        gpu_energy_joules=gpu_energy_j,
        rapl_energy_joules=rapl_energy_j,
        total_energy_joules=total_energy,
        joules_per_token=jpt,
        joules_per_request=jpr,
        duration_s=duration_s,
    )


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _aggregate_gpu(gpu_series: list[GPUSample]) -> tuple[float, float, float]:
    """Return (avg_w, peak_w, energy_joules) summed across all visible GPUs.

    Energy is computed by trapezoidal integration of the per-sample total
    power (sum across devices) over time.
    """
    if not gpu_series:
        return 0.0, 0.0, 0.0

    times_s: list[float] = []
    powers_w: list[float] = []
    peak_w = 0.0
    for s in gpu_series:
        total_w = 0.0
        for d in s.devices:
            p = d.get("power_w")
            if isinstance(p, int | float):
                total_w += float(p)
        times_s.append(s.t_ms / 1000.0)
        powers_w.append(total_w)
        peak_w = max(peak_w, total_w)

    if not powers_w:
        return 0.0, 0.0, 0.0

    avg_w = sum(powers_w) / len(powers_w)
    energy_j = _trapz(times_s, powers_w)
    return avg_w, peak_w, energy_j


def _aggregate_rapl(rapl_series: list[RAPLSample]) -> float:
    """Return total RAPL energy delta across all observed domains, in joules.

    RAPL counters are monotonic microjoules. Energy = (last - first) / 1e6.
    """
    if len(rapl_series) < 2:
        return 0.0

    # Build first/last energy per domain name
    first: dict[str, int] = {}
    last: dict[str, int] = {}
    for sample in rapl_series:
        for d in sample.domains:
            name = str(d.get("name", ""))
            energy = int(d.get("energy_uj", 0))
            if name not in first:
                first[name] = energy
            last[name] = energy

    total_j = 0.0
    for name, end in last.items():
        start = first.get(name, end)
        delta_uj = end - start
        # RAPL counters wrap on overflow; if delta is negative treat as 0
        if delta_uj < 0:
            continue
        total_j += delta_uj / 1e6
    return total_j


def _trapz(xs: list[float], ys: list[float]) -> float:
    """Trapezoidal integration. xs and ys must be the same length, xs sorted."""
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    total = 0.0
    for i in range(1, len(xs)):
        dx = xs[i] - xs[i - 1]
        if dx <= 0:
            continue
        total += dx * (ys[i] + ys[i - 1]) / 2.0
    return total
