"""Base classes for telemetry samplers."""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from types import TracebackType


@dataclass(frozen=True, slots=True)
class TelemetrySample:
    """Minimum shape every sampler exposes: timestamp + a per-vendor payload.

    Concrete samplers extend with vendor-specific fields (GPU temp, RAPL energy).
    The :attr:`t_ms` is wall-clock since sampler start, in milliseconds —
    a monotonic clock so we can correlate samplers with each other and with
    request samples from the drivers.
    """

    t_ms: float


class Sampler(ABC):
    """Background-thread sampler abstract base.

    Subclasses implement :meth:`_one_sample` (single read) and the base class
    handles the polling loop, start/stop, and snapshot extraction.

    Use as a context manager — start on enter, stop on exit::

        with NVMLSampler(50) as s:
            ... do work ...
        rows = s.snapshot()
    """

    def __init__(self, interval_ms: int) -> None:
        if interval_ms < 1:
            msg = "interval_ms must be ≥ 1"
            raise ValueError(msg)
        self._interval_s = interval_ms / 1000.0
        self._samples: list[TelemetrySample] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._t0 = 0.0

    # ------------------------------------------------------------------- API
    def start(self) -> None:
        """Begin polling in a background thread. Idempotent if already started."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._samples = []
        self._stop.clear()
        self._t0 = time.perf_counter()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the background thread and join. Idempotent."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def snapshot(self) -> list[TelemetrySample]:
        """Return a copy of the samples collected since :meth:`start`."""
        return list(self._samples)

    # ------------------------------------------------------------- protocol
    @abstractmethod
    def _one_sample(self, t_ms: float) -> TelemetrySample | None:
        """Take ONE reading. Return the sample, or None to skip this tick.

        Subclasses must NOT raise — silent skips are OK, hard failures
        should be raised in :meth:`_setup` (called once before polling).
        """

    def _setup(self) -> None:
        """Run once before polling starts. Subclasses can override (e.g. nvmlInit)."""

    def _teardown(self) -> None:
        """Run once after polling stops. Subclasses can override (e.g. nvmlShutdown)."""

    # --------------------------------------------------------------- impl
    def _loop(self) -> None:
        try:
            self._setup()
        except Exception:
            return
        try:
            while not self._stop.is_set():
                t_ms = (time.perf_counter() - self._t0) * 1000.0
                sample = self._one_sample(t_ms)
                if sample is not None:
                    self._samples.append(sample)
                # Sleep with stop-event awareness for prompt shutdown
                if self._stop.wait(timeout=self._interval_s):
                    break
        finally:
            self._teardown()

    # ----------------------------------------------------- context manager
    def __enter__(self) -> Sampler:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.stop()
