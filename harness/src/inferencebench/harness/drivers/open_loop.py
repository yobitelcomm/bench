"""Open-loop driver with Poisson arrivals.

Used when we want to characterise the service curve at a target RPS — i.e.
how the system responds to a workload that arrives independently of how long
prior requests took.

Each request is scheduled at ``t_n = t_{n-1} + Exp(1/lambda)`` where
``lambda = mean_rps``. Inter-arrival times follow an exponential distribution,
which yields a Poisson arrival process — the standard model for independent
arrivals (M/M/c queues etc.).

If the system can't keep up, requests pile up; we cap concurrent in-flight
requests with ``max_inflight`` to avoid unbounded memory growth.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from inferencebench.harness.drivers.base import RequestFn, Sample


@dataclass(slots=True)
class OpenLoopDriver:
    """Drive requests at a target mean RPS with Poisson inter-arrivals.

    Args:
        mean_rps: Target arrival rate (requests per second).
        duration_s: Measurement window in seconds.
        seed: PRNG seed for reproducible arrival times.
        max_inflight: Hard cap on simultaneously-in-flight requests; protects
            against runaway queues if the system is overloaded. Defaults to
            ``8 * mean_rps`` (≥8 in-flight per second of expected load).
        warmup_requests: Discard the first N completed requests. The harness's
            convergence gate handles steady-state detection separately.
    """

    mean_rps: float
    duration_s: float
    seed: int = 42
    max_inflight: int | None = None
    warmup_requests: int = 0
    _samples: list[Sample] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.mean_rps <= 0:
            msg = "mean_rps must be positive"
            raise ValueError(msg)
        if self.duration_s <= 0:
            msg = "duration_s must be positive"
            raise ValueError(msg)
        if self.max_inflight is None:
            self.max_inflight = max(8, int(self.mean_rps * 8))

    def execute(self, request_fn: RequestFn, workload: list[Any]) -> list[Sample]:
        """Run the driver. Returns the full Sample list (post-warmup discard).

        Args:
            request_fn: Function ``(request_idx, item) -> Sample`` to call for each request.
            workload: List of items to draw from (round-robin). Empty list raises.
        """
        if not workload:
            msg = "workload must contain at least one item"
            raise ValueError(msg)

        rng = np.random.default_rng(self.seed)
        # Pre-generate inter-arrival times for the whole window so we can cleanly
        # bound execution to `duration_s` rather than draw lazily.
        # Over-generate by 2x to handle stochastic overshoot.
        expected_n = max(int(self.mean_rps * self.duration_s * 2), 16)
        inter_arrivals = rng.exponential(scale=1.0 / self.mean_rps, size=expected_n)
        arrivals = np.cumsum(inter_arrivals)

        # Truncate to the measurement window
        arrivals = arrivals[arrivals < self.duration_s]
        if len(arrivals) == 0:
            return []

        results: list[Sample | None] = [None] * len(arrivals)
        in_flight = threading.Semaphore(self.max_inflight or 1)
        lock = threading.Lock()

        def _wrapper(idx: int, scheduled_at: float, item: Any) -> None:
            # Block if we hit the in-flight cap (simulates a finite queue)
            in_flight.acquire()
            try:
                start_ms = (time.perf_counter() - run_t0) * 1000.0
                try:
                    sample = request_fn(idx, item)
                except Exception as exc:
                    sample = Sample(
                        request_idx=idx,
                        arrival_ms=scheduled_at,
                        start_ms=start_ms,
                        ttft_ms=float("nan"),
                        total_ms=float("nan"),
                        tpot_ms=float("nan"),
                        tokens_in=0,
                        tokens_out=0,
                        cost_usd=0.0,
                        finish_reason="error",
                        ok=False,
                        error=str(exc),
                    )
                # Always overwrite arrival/start (caller may have produced its own)
                sample = Sample(
                    request_idx=idx,
                    arrival_ms=scheduled_at,
                    start_ms=start_ms,
                    ttft_ms=sample.ttft_ms,
                    total_ms=sample.total_ms,
                    tpot_ms=sample.tpot_ms,
                    tokens_in=sample.tokens_in,
                    tokens_out=sample.tokens_out,
                    cost_usd=sample.cost_usd,
                    finish_reason=sample.finish_reason,
                    ok=sample.ok,
                    error=sample.error,
                    extra=sample.extra,
                )
                with lock:
                    results[idx] = sample
            finally:
                in_flight.release()

        run_t0 = time.perf_counter()

        # Bounded thread pool — at most max_inflight concurrent — protects against
        # over-provisioning the executor when the engine is slow.
        with ThreadPoolExecutor(max_workers=self.max_inflight or 1) as pool:
            futures: list[Future[None]] = []
            for i, scheduled_at_s in enumerate(arrivals):
                # Sleep until the next scheduled arrival
                target_t = run_t0 + scheduled_at_s
                now = time.perf_counter()
                if target_t > now:
                    time.sleep(target_t - now)
                scheduled_ms = scheduled_at_s * 1000.0
                item = workload[i % len(workload)]
                futures.append(pool.submit(_wrapper, i, scheduled_ms, item))
            for f in futures:
                f.result()

        # Filter out warmup + None slots (None shouldn't happen but be defensive)
        completed = [s for s in results if s is not None]
        return completed[self.warmup_requests :]
