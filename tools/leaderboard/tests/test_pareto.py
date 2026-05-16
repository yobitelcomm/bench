"""Synthetic-data tests for compute_pareto."""

from __future__ import annotations

from inferencebench_leaderboard import compute_pareto


def test_pareto_max_min_default() -> None:
    # Points: (throughput, latency). Higher throughput, lower latency = better.
    #   A: (100, 10)   — on frontier (best throughput at tied-best latency-ish)
    #   B: (90, 5)     — on frontier (best latency)
    #   C: (50, 20)    — dominated by both A and B
    #   D: (100, 20)   — dominated by A (same throughput, worse latency)
    entries = [(100.0, 10.0), (90.0, 5.0), (50.0, 20.0), (100.0, 20.0)]
    flags = compute_pareto(entries)
    assert flags == [True, True, False, False]


def test_pareto_all_min() -> None:
    # Both axes minimize. Frontier = bottom-left non-dominated.
    entries = [(1.0, 5.0), (2.0, 2.0), (5.0, 1.0), (10.0, 10.0)]
    flags = compute_pareto(entries, x_direction="min", y_direction="min")
    assert flags == [True, True, True, False]


def test_pareto_all_max() -> None:
    entries = [(1.0, 5.0), (2.0, 2.0), (5.0, 1.0), (10.0, 10.0)]
    flags = compute_pareto(entries, x_direction="max", y_direction="max")
    # Only (10,10) dominates all the others.
    assert flags == [False, False, False, True]


def test_pareto_singleton_is_on_frontier() -> None:
    assert compute_pareto([(1.0, 1.0)]) == [True]


def test_pareto_empty() -> None:
    assert compute_pareto([]) == []


def test_pareto_missing_values_marked_false() -> None:
    entries = [(100.0, 10.0), (None, 5.0), (90.0, None), (None, None)]
    flags = compute_pareto(entries)
    assert flags[0] is True
    assert flags[1] is False
    assert flags[2] is False
    assert flags[3] is False


def test_pareto_strict_domination_required() -> None:
    # Two identical points: neither strictly dominates the other, so both stay.
    entries = [(10.0, 5.0), (10.0, 5.0)]
    flags = compute_pareto(entries)
    assert flags == [True, True]
