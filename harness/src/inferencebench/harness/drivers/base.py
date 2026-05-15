"""Shared types for harness drivers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class Sample:
    """One observation from a single request.

    Drivers fill these in; downstream code aggregates them into percentiles
    and SLO compliance numbers.

    Times are in milliseconds (perf_counter precision). ``ok=False`` samples
    are excluded from latency stats but counted toward error-rate metrics.
    """

    request_idx: int
    arrival_ms: float  # wall-clock since run start when this request was scheduled
    start_ms: float  # wall-clock when the request actually fired (queue delay = start-arrival)
    ttft_ms: float  # time-to-first-token, ms
    total_ms: float  # total request latency, ms
    tpot_ms: float  # average time-per-output-token (excluding TTFT)
    tokens_in: int
    tokens_out: int
    cost_usd: float
    finish_reason: str
    ok: bool  # False if the request errored
    error: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


# Callable signature drivers expect: takes a workload item, returns a Sample.
# Drivers don't know what a "workload item" is — that's the plugin's call.
# The function does I/O (model invocation) and returns timing info as a Sample.
RequestFn = Callable[[int, Any], Sample]
