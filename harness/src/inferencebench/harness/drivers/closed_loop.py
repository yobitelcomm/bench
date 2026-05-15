"""Closed-loop driver with bounded concurrency.

N workers each run requests back-to-back: each worker waits for its previous
request to complete before issuing the next. The total request rate is
self-limited by the system's service time — no queue builds up.

Use this for engine vs engine comparisons where you want to isolate raw
service speed without the noise of arrival-pattern dependencies.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

from inferencebench.harness.drivers.base import RequestFn, Sample


@dataclass(slots=True)
class ClosedLoopDriver:
    """Run ``concurrency`` workers in parallel, each looping back-to-back requests.

    Args:
        concurrency: Number of parallel workers. Each is its own thread.
        duration_s: Measurement window. Workers stop once it elapses.
        warmup_requests: Discard the first N completed requests (across all workers).
        max_requests: Optional hard cap on completed requests (per worker * workers).
    """

    concurrency: int
    duration_s: float
    warmup_requests: int = 0
    max_requests: int | None = None
    _samples: list[Sample] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.concurrency < 1:
            msg = "concurrency must be ≥ 1"
            raise ValueError(msg)
        if self.duration_s <= 0:
            msg = "duration_s must be positive"
            raise ValueError(msg)

    def execute(self, request_fn: RequestFn, workload: list[Any]) -> list[Sample]:
        """Run the driver. Returns the Sample list (post-warmup discard)."""
        if not workload:
            msg = "workload must contain at least one item"
            raise ValueError(msg)

        results: list[Sample] = []
        results_lock = threading.Lock()
        stop_event = threading.Event()
        idx_counter = [0]
        idx_lock = threading.Lock()

        run_t0 = time.perf_counter()

        def _worker(worker_id: int) -> None:
            local_idx = 0
            while not stop_event.is_set():
                # Bail out if we hit the time bound
                if (time.perf_counter() - run_t0) >= self.duration_s:
                    return

                with idx_lock:
                    global_idx = idx_counter[0]
                    idx_counter[0] += 1
                    if self.max_requests is not None and global_idx >= self.max_requests:
                        return

                item = workload[global_idx % len(workload)]
                scheduled_ms = (time.perf_counter() - run_t0) * 1000.0
                start_ms = scheduled_ms  # closed-loop: arrival == start (no queue)

                try:
                    sample = request_fn(global_idx, item)
                except Exception as exc:
                    sample = Sample(
                        request_idx=global_idx,
                        arrival_ms=scheduled_ms,
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

                # Replace arrival/start with closed-loop semantics (no queue delay)
                sample = Sample(
                    request_idx=global_idx,
                    arrival_ms=scheduled_ms,
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
                    extra={**sample.extra, "worker_id": worker_id},
                )
                with results_lock:
                    results.append(sample)
                local_idx += 1

        # Start workers
        workers = [
            threading.Thread(target=_worker, args=(i,), daemon=True)
            for i in range(self.concurrency)
        ]
        for w in workers:
            w.start()

        # Wait the measurement window, then signal stop
        time.sleep(self.duration_s)
        stop_event.set()
        for w in workers:
            w.join(timeout=10.0)

        # Sort by completion order (request_idx) for determinism + drop warmup
        results.sort(key=lambda s: s.request_idx)
        return results[self.warmup_requests :]
