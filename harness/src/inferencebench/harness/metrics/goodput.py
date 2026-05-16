"""Goodput-at-SLO: requests-per-second that met all configured SLOs.

This is the single most useful headline number for procurement / capacity
planning. Throughput tells you what the system can do at peak; goodput tells
you what fraction of that throughput is actually useful (meets latency,
quality, cost constraints).

SLOs are specified as a list of :class:`SLOPredicate` — each one is a
named threshold against a Sample field (e.g. ``ttft_ms < 200``).
A Sample passes if it satisfies ALL predicates AND has ``ok=True``.

Public API::

    from inferencebench.harness.metrics import GoodputAtSLO, SLOPredicate

    slos = [
        SLOPredicate("ttft", field="ttft_ms", op="<", value=200.0),
        SLOPredicate("tpot", field="tpot_ms", op="<", value=50.0),
        SLOPredicate("total", field="total_ms", op="<", value=3000.0),
    ]
    goodput = GoodputAtSLO(samples, duration_s=300.0, slos=slos)
    print(goodput.req_per_s_passing, goodput.compliance_rate)
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from inferencebench.harness.drivers import Sample

SLOOp = Literal["<", "<=", ">", ">=", "=="]


@dataclass(frozen=True, slots=True)
class SLOPredicate:
    """One named SLO threshold.

    Args:
        name: Identifier for reports (e.g. ``"ttft"``).
        field: Sample attribute to test (e.g. ``"ttft_ms"``).
        op: Comparison operator: ``"<"``, ``"<="``, ``">"``, ``">="``, ``"=="``.
        value: Threshold to compare against.
    """

    name: str
    field: str
    op: SLOOp
    value: float

    def evaluate(self, sample: Sample) -> bool:
        """Return True if this predicate passes on ``sample``. NaN/missing → False."""
        try:
            v = getattr(sample, self.field)
        except AttributeError:
            return False
        try:
            v = float(v)
        except (TypeError, ValueError):
            return False
        if v != v:  # NaN
            return False
        if self.op == "<":
            return v < self.value
        if self.op == "<=":
            return v <= self.value
        if self.op == ">":
            return v > self.value
        if self.op == ">=":
            return v >= self.value
        return v == self.value


@dataclass(frozen=True, slots=True)
class GoodputAtSLO:
    """Aggregate goodput numbers for a sample stream against SLOs.

    Args:
        samples: Iterable of :class:`Sample` objects from a driver run.
        duration_s: Wall-clock measurement window. Required for req/s.
        slos: List of :class:`SLOPredicate`. Empty list ⇒ ``passing == ok``.

    Computed fields:
        total_samples, ok_samples, passing_samples, failing_samples,
        compliance_rate, req_per_s_passing, req_per_s_all, per_slo_pass_rate.
    """

    total_samples: int
    ok_samples: int
    passing_samples: int
    failing_samples: int
    compliance_rate: float  # passing / total
    ok_rate: float  # ok / total (excludes errors)
    req_per_s_passing: float  # passing / duration_s
    req_per_s_all: float  # total / duration_s
    per_slo_pass_rate: dict[str, float]
    duration_s: float
    slo_names: tuple[str, ...]

    @classmethod
    def from_samples(
        cls,
        samples: Iterable[Sample],
        *,
        duration_s: float,
        slos: list[SLOPredicate] | None = None,
    ) -> GoodputAtSLO:
        """Build a :class:`GoodputAtSLO` from a sample stream + SLOs.

        Raises:
            ValueError: ``duration_s`` must be positive.
        """
        if duration_s <= 0:
            msg = "duration_s must be positive"
            raise ValueError(msg)

        slos = list(slos or [])
        sample_list = list(samples)
        total = len(sample_list)
        ok_count = sum(1 for s in sample_list if s.ok)

        per_slo_pass = {p.name: 0 for p in slos}
        passing = 0
        for s in sample_list:
            if not s.ok:
                continue
            all_pass = True
            for p in slos:
                if p.evaluate(s):
                    per_slo_pass[p.name] += 1
                else:
                    all_pass = False
            if all_pass:
                passing += 1

        failing = total - passing
        compliance_rate = (passing / total) if total else 0.0
        ok_rate = (ok_count / total) if total else 0.0

        # Per-SLO pass rate is over OK samples (errors aren't an SLO miss)
        per_slo_rate: dict[str, float] = {
            name: (count / ok_count if ok_count else 0.0) for name, count in per_slo_pass.items()
        }

        return cls(
            total_samples=total,
            ok_samples=ok_count,
            passing_samples=passing,
            failing_samples=failing,
            compliance_rate=compliance_rate,
            ok_rate=ok_rate,
            req_per_s_passing=passing / duration_s,
            req_per_s_all=total / duration_s,
            per_slo_pass_rate=per_slo_rate,
            duration_s=duration_s,
            slo_names=tuple(p.name for p in slos),
        )

    def as_dict(self) -> dict[str, float | int | dict[str, float]]:
        """Flat-ish dict for embedding in envelopes / JSON."""
        return {
            "total_samples": self.total_samples,
            "ok_samples": self.ok_samples,
            "passing_samples": self.passing_samples,
            "failing_samples": self.failing_samples,
            "compliance_rate": self.compliance_rate,
            "ok_rate": self.ok_rate,
            "req_per_s_passing": self.req_per_s_passing,
            "req_per_s_all": self.req_per_s_all,
            "duration_s": self.duration_s,
            "per_slo_pass_rate": dict(self.per_slo_pass_rate),
        }
