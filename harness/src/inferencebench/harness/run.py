"""BenchmarkRun — top-level orchestrator that ties drivers + telemetry + metrics together.

This is the bridge between a per-modality plugin and the harness primitives.
A plugin builds a :class:`BenchmarkRun`, calls :meth:`execute`, and gets back
a fully-populated :class:`RunResult` containing:

- Per-request samples (latency timeseries)
- Telemetry snapshots (NVML + RAPL series)
- Aggregated metrics (Percentiles, GoodputAtSLO)
- Convergence state (did we reach steady state?)

The plugin then wraps the RunResult into a signed Envelope.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from inferencebench.harness.convergence import ConvergenceGate, ConvergenceState
from inferencebench.harness.drivers import (
    ClosedLoopDriver,
    OpenLoopDriver,
    Sample,
)
from inferencebench.harness.metrics import GoodputAtSLO, Percentiles, SLOPredicate
from inferencebench.harness.telemetry import GPUSample, NVMLSampler, RAPLSample, RAPLSampler


@dataclass(frozen=True, slots=True)
class RunResult:
    """All the raw + aggregated outputs from one benchmark run."""

    samples: list[Sample]
    gpu_telemetry: list[GPUSample]
    rapl_telemetry: list[RAPLSample]
    convergence: ConvergenceState
    duration_s: float

    # Aggregates (computed lazily by the helpers below)
    ttft_percentiles: Percentiles | None = None
    tpot_percentiles: Percentiles | None = None
    total_percentiles: Percentiles | None = None
    goodput: GoodputAtSLO | None = None

    # Plugin-attached metadata
    extra: dict[str, Any] = field(default_factory=dict)

    def compute_metrics(self, slos: list[SLOPredicate] | None = None) -> RunResult:
        """Return a new RunResult with all Percentiles + GoodputAtSLO filled in."""
        ttft = [s.ttft_ms for s in self.samples if s.ok]
        tpot = [s.tpot_ms for s in self.samples if s.ok]
        total = [s.total_ms for s in self.samples if s.ok]

        ttft_p = Percentiles(ttft) if ttft else None
        tpot_p = Percentiles(tpot) if tpot else None
        total_p = Percentiles(total) if total else None
        gp = GoodputAtSLO.from_samples(self.samples, duration_s=self.duration_s, slos=slos)

        return RunResult(
            samples=self.samples,
            gpu_telemetry=self.gpu_telemetry,
            rapl_telemetry=self.rapl_telemetry,
            convergence=self.convergence,
            duration_s=self.duration_s,
            ttft_percentiles=ttft_p,
            tpot_percentiles=tpot_p,
            total_percentiles=total_p,
            goodput=gp,
            extra=self.extra,
        )


DriverConfig = OpenLoopDriver | ClosedLoopDriver


@dataclass(slots=True)
class BenchmarkRun:
    """One execution of a benchmark: driver + samplers + convergence gate.

    Args:
        driver: Configured :class:`OpenLoopDriver` or :class:`ClosedLoopDriver`.
        workload: List of items the driver round-robins through. Plugin-defined.
        request_fn: Callback ``(idx, item) -> Sample``. Plugin wraps its
            engine client here.
        nvml_interval_ms: GPU telemetry interval. 0 disables.
        rapl_interval_ms: CPU/DRAM telemetry interval. 0 disables.
        convergence: Optional pre-configured convergence gate. None disables.
    """

    driver: DriverConfig
    workload: list[Any]
    request_fn: Callable[[int, Any], Sample]
    nvml_interval_ms: int = 50
    rapl_interval_ms: int = 100
    convergence: ConvergenceGate | None = None

    def execute(self) -> RunResult:
        """Run driver + samplers + gate; return :class:`RunResult`."""
        nvml = NVMLSampler(self.nvml_interval_ms) if self.nvml_interval_ms > 0 else None
        rapl = RAPLSampler(self.rapl_interval_ms) if self.rapl_interval_ms > 0 else None

        t0 = time.perf_counter()
        if nvml is not None:
            nvml.start()
        if rapl is not None:
            rapl.start()

        try:
            # Wrap the user's request_fn to drive the convergence gate
            gate = self.convergence
            user_fn = self.request_fn

            def gated_fn(idx: int, item: Any) -> Sample:
                sample = user_fn(idx, item)
                if gate is not None and sample.ok:
                    gate.update(sample.total_ms)
                return sample

            samples = self.driver.execute(gated_fn, self.workload)
        finally:
            if nvml is not None:
                nvml.stop()
            if rapl is not None:
                rapl.stop()

        duration_s = time.perf_counter() - t0

        gpu_series: list[GPUSample] = (
            [s for s in nvml.snapshot() if isinstance(s, GPUSample)] if nvml is not None else []
        )
        rapl_series: list[RAPLSample] = (
            [s for s in rapl.snapshot() if isinstance(s, RAPLSample)] if rapl is not None else []
        )
        conv_state = (
            self.convergence.state
            if self.convergence
            else ConvergenceState(
                n_seen=0,
                n_warmed=0,
                cov=float("nan"),
                converged=False,
                bailed_out=False,
            )
        )

        return RunResult(
            samples=samples,
            gpu_telemetry=gpu_series,
            rapl_telemetry=rapl_series,
            convergence=conv_state,
            duration_s=duration_s,
        )
