"""Telemetry samplers — background pollers that record time-series device state.

Phase 1 ships :class:`NVMLSampler` (NVIDIA GPU) and :class:`RAPLSampler`
(Intel CPU + DRAM via `/sys/class/powercap`). Both implement the
:class:`Sampler` protocol so harness code can compose them without knowing
the vendor.

Usage::

    from inferencebench.harness.telemetry import NVMLSampler, RAPLSampler

    with NVMLSampler(interval_ms=50) as gpu, RAPLSampler(interval_ms=100) as cpu:
        run_benchmark_workload()
    gpu_series = gpu.snapshot()  # list[GPUSample]
    cpu_series = cpu.snapshot()  # list[RAPLSample]
"""

from inferencebench.harness.telemetry.base import Sampler, TelemetrySample
from inferencebench.harness.telemetry.nvml import GPUSample, NVMLSampler
from inferencebench.harness.telemetry.rapl import RAPLSample, RAPLSampler

__all__ = [
    "GPUSample",
    "NVMLSampler",
    "RAPLSample",
    "RAPLSampler",
    "Sampler",
    "TelemetrySample",
]
