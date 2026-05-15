"""NVIDIA GPU telemetry via NVML.

Polls each device for utilization, memory, power, temperature, and clock state
at a fixed interval. Designed to add <1% overhead at 50 ms intervals on modern
NVIDIA GPUs (the NVML calls take ~50-200 μs each).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from inferencebench.harness.telemetry.base import Sampler, TelemetrySample


@dataclass(frozen=True, slots=True)
class GPUSample(TelemetrySample):
    """One reading across all visible NVIDIA GPUs."""

    devices: tuple[dict[str, float | int], ...]
    # Each device dict carries: gpu_index, util_gpu_pct, util_mem_pct,
    # mem_used_mb, mem_total_mb, power_w, temp_c, sm_clock_mhz,
    # mem_clock_mhz, throttle_reasons.


class NVMLSampler(Sampler):
    """Poll NVML at a fixed interval. No-op if pynvml or driver is absent.

    Args:
        interval_ms: Polling period in milliseconds. 50 ms is a good default
            for benchmark runs; 25 ms for TTFT-critical work; 200 ms for
            long-running offline sweeps.
        device_indices: Optional restriction to specific GPU indices. None = all.
    """

    def __init__(self, interval_ms: int = 50, *, device_indices: list[int] | None = None) -> None:
        super().__init__(interval_ms=interval_ms)
        self._device_indices = device_indices
        self._nvml: Any = None
        self._handles: list[Any] = []

    def _setup(self) -> None:
        try:
            import pynvml
        except ImportError:
            self._nvml = None
            return
        try:
            pynvml.nvmlInit()
        except pynvml.NVMLError:
            self._nvml = None
            return
        self._nvml = pynvml
        try:
            n = pynvml.nvmlDeviceGetCount()
        except pynvml.NVMLError:
            self._handles = []
            return
        indices = self._device_indices if self._device_indices is not None else list(range(n))
        self._handles = []
        for i in indices:
            try:
                self._handles.append(pynvml.nvmlDeviceGetHandleByIndex(i))
            except pynvml.NVMLError:
                continue

    def _teardown(self) -> None:
        if self._nvml is not None:
            try:
                self._nvml.nvmlShutdown()
            except self._nvml.NVMLError:
                pass

    def _one_sample(self, t_ms: float) -> GPUSample | None:
        if self._nvml is None or not self._handles:
            return None
        nvml = self._nvml
        devices: list[dict[str, float | int]] = []
        for idx, handle in enumerate(self._handles):
            d: dict[str, float | int] = {"gpu_index": idx}
            try:
                util = nvml.nvmlDeviceGetUtilizationRates(handle)
                d["util_gpu_pct"] = int(util.gpu)
                d["util_mem_pct"] = int(util.memory)
            except nvml.NVMLError:
                pass
            try:
                mem = nvml.nvmlDeviceGetMemoryInfo(handle)
                d["mem_used_mb"] = int(mem.used // (1024 * 1024))
                d["mem_total_mb"] = int(mem.total // (1024 * 1024))
            except nvml.NVMLError:
                pass
            try:
                d["power_w"] = nvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
            except nvml.NVMLError:
                pass
            try:
                d["temp_c"] = int(nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU))
            except nvml.NVMLError:
                pass
            try:
                d["sm_clock_mhz"] = int(nvml.nvmlDeviceGetClockInfo(handle, nvml.NVML_CLOCK_SM))
                d["mem_clock_mhz"] = int(nvml.nvmlDeviceGetClockInfo(handle, nvml.NVML_CLOCK_MEM))
            except nvml.NVMLError:
                pass
            try:
                d["throttle_reasons"] = int(nvml.nvmlDeviceGetCurrentClocksThrottleReasons(handle))
            except nvml.NVMLError:
                pass
            devices.append(d)
        return GPUSample(t_ms=t_ms, devices=tuple(devices))
