"""Request drivers for benchmark workloads.

A driver feeds requests at a controlled rate into a :class:`ModelClient` and
collects per-request timing samples. Drivers are reusable across plugins and
modalities (LLM, voice, embedding) — the workload changes, the driving pattern
doesn't.

Phase 1 ships:

* :class:`OpenLoopDriver` — Poisson arrivals at a fixed mean RPS. Captures the
  service curve under load; the canonical "what's the throughput at SLO" driver.
* :class:`ClosedLoopDriver` — bounded concurrency (N parallel workers). Captures
  steady-state behaviour without a queue; ideal for engine comparisons.

Both produce :class:`Sample` lists that downstream metrics code turns into
percentiles, goodput, RTF, etc.
"""

from inferencebench.harness.drivers.base import RequestFn, Sample
from inferencebench.harness.drivers.closed_loop import ClosedLoopDriver
from inferencebench.harness.drivers.open_loop import OpenLoopDriver

__all__ = [
    "ClosedLoopDriver",
    "OpenLoopDriver",
    "RequestFn",
    "Sample",
]
