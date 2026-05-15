"""Warmup + convergence gate.

Every measurement-grade benchmark MUST pass through this gate:

1. Discard the first ``warmup_runs`` samples (engine warm-up, cache fill).
2. Run requests until the coefficient of variation (CoV = stdev / mean)
   over the last ``window`` samples drops below ``threshold``.
3. Once stable, the *measurement window* begins — subsequent samples count.

CoV is a unitless dispersion measure; CoV < 0.05 (5%) is the usual benchmark
threshold for "steady state".

This module is engine-agnostic and stream-friendly: feed samples one at a time
to :meth:`update`, ask :attr:`is_converged` between requests, stop driving when
``True``.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

DEFAULT_WARMUP_RUNS = 3
DEFAULT_WINDOW = 30
DEFAULT_THRESHOLD = 0.05  # 5% CoV
DEFAULT_MAX_WAIT_REQUESTS = 500  # bail out if we never converge


@dataclass(frozen=True, slots=True)
class ConvergenceState:
    """Snapshot of the gate's state for logging / debugging."""

    n_seen: int  # total samples passed to update()
    n_warmed: int  # samples remaining after discarding warmup
    cov: float  # current CoV across the rolling window (NaN if window not full)
    converged: bool
    bailed_out: bool  # True if max_wait_requests hit without convergence


class ConvergenceGate:
    """Stream-fed warmup + CoV gate.

    Args:
        warmup_runs: Discard this many initial samples unconditionally.
        window: Rolling window size for CoV computation. Must be ≥ 2.
        threshold: CoV must be < this to declare convergence (default 0.05).
        max_wait_requests: After this many post-warmup samples without
            convergence, set :attr:`bailed_out=True`; callers should report
            non-convergence as a benchmark warning, not a crash.

    Typical usage::

        gate = ConvergenceGate()
        for sample in stream:
            gate.update(sample.total_ms)
            if gate.is_converged:
                break
        if gate.state.bailed_out:
            warnings.append("convergence-not-reached")
    """

    def __init__(
        self,
        *,
        warmup_runs: int = DEFAULT_WARMUP_RUNS,
        window: int = DEFAULT_WINDOW,
        threshold: float = DEFAULT_THRESHOLD,
        max_wait_requests: int = DEFAULT_MAX_WAIT_REQUESTS,
    ) -> None:
        if warmup_runs < 0:
            msg = "warmup_runs must be ≥ 0"
            raise ValueError(msg)
        if window < 2:
            msg = "window must be ≥ 2"
            raise ValueError(msg)
        if threshold <= 0:
            msg = "threshold must be > 0"
            raise ValueError(msg)

        self.warmup_runs = warmup_runs
        self.window = window
        self.threshold = threshold
        self.max_wait_requests = max_wait_requests

        self._n_seen = 0
        self._n_warmed = 0
        self._buf: deque[float] = deque(maxlen=window)
        self._converged = False
        self._bailed_out = False
        self._cov = float("nan")

    def update(self, value: float) -> None:
        """Feed one new sample. Calling order: warmup → window-fill → convergence-check."""
        self._n_seen += 1
        if not np.isfinite(value):
            return
        if self._n_seen <= self.warmup_runs:
            return
        self._n_warmed += 1
        self._buf.append(float(value))

        if len(self._buf) < self.window:
            return  # window not yet full

        arr = np.asarray(self._buf, dtype=np.float64)
        mean = arr.mean()
        if mean <= 0:
            self._cov = float("inf")
        else:
            self._cov = float(arr.std(ddof=1) / mean)

        if self._cov < self.threshold:
            self._converged = True
        elif self._n_warmed >= self.max_wait_requests:
            self._bailed_out = True

    @property
    def is_converged(self) -> bool:
        return self._converged

    @property
    def bailed_out(self) -> bool:
        return self._bailed_out

    @property
    def state(self) -> ConvergenceState:
        return ConvergenceState(
            n_seen=self._n_seen,
            n_warmed=self._n_warmed,
            cov=self._cov,
            converged=self._converged,
            bailed_out=self._bailed_out,
        )

    def reset(self) -> None:
        """Reset all state. Useful for multiple measurement windows in one run."""
        self._n_seen = 0
        self._n_warmed = 0
        self._buf.clear()
        self._converged = False
        self._bailed_out = False
        self._cov = float("nan")
