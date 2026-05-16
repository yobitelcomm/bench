"""``bench compare`` — compare benchmark runs, render Pareto frontier.

Phase 1 implementation (ticket 0026). Loads 2+ signed envelope JSON files from
local paths, computes Pareto frontiers across the canonical metric pairs, and
renders them as a Rich table or JSON.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from inferencebench.envelope import Envelope, verify_envelope

console = Console()
err_console = Console(stderr=True)


# --------------------------------------------------------------------------- #
# Pareto helper                                                               #
# --------------------------------------------------------------------------- #
# Canonical metric pairs we compare across. Each tuple is
# ``(label, x_metric, y_metric, x_maximize, y_maximize)``.
_METRIC_PAIRS: list[tuple[str, str, str, bool, bool]] = [
    # Quality vs Cost: prefer higher goodput, lower cost.
    ("quality_vs_cost", "goodput_at_slo", "cost_usd_per_million_tokens", True, False),
    # Throughput vs Latency: prefer higher throughput, lower latency.
    ("throughput_vs_latency", "throughput_tok_per_s", "ttft_p99_ms", True, False),
    # Throughput vs Energy: prefer higher throughput, lower energy.
    ("throughput_vs_energy", "throughput_tok_per_s", "joules_per_token", True, False),
]


def _pareto_front(
    points: list[tuple[float | None, float | None]],
    *,
    maximize_x: bool,
    maximize_y: bool,
) -> list[bool]:
    """Classify each ``(x, y)`` point as on the Pareto frontier or dominated.

    Points with ``None`` on either axis are treated as missing and are never on
    the frontier. Uses the same dominance semantics as
    :func:`inferencebench_leaderboard.compute_pareto` but inlined to avoid a
    hard CLI -> leaderboard dependency.

    Args:
        points: ``(x, y)`` coordinates, ``None`` denotes a missing value.
        maximize_x: ``True`` if higher x is better.
        maximize_y: ``True`` if higher y is better.

    Returns:
        Boolean flags, one per input, ``True`` iff the point is on the frontier.
    """

    def better_or_equal(a: float, b: float, *, maximize: bool) -> bool:
        return a >= b if maximize else a <= b

    def strictly_better(a: float, b: float, *, maximize: bool) -> bool:
        return a > b if maximize else a < b

    result = [False] * len(points)
    for i, (xi, yi) in enumerate(points):
        if xi is None or yi is None:
            continue
        dominated = False
        for j, (xj, yj) in enumerate(points):
            if i == j or xj is None or yj is None:
                continue
            if (
                better_or_equal(xj, xi, maximize=maximize_x)
                and better_or_equal(yj, yi, maximize=maximize_y)
                and (
                    strictly_better(xj, xi, maximize=maximize_x)
                    or strictly_better(yj, yi, maximize=maximize_y)
                )
            ):
                dominated = True
                break
        result[i] = not dominated
    return result


# --------------------------------------------------------------------------- #
# Command                                                                     #
# --------------------------------------------------------------------------- #
def compare(
    run_ids: Annotated[
        list[str],
        typer.Argument(help="Two or more envelope file paths to compare (local paths only)."),
    ],
    report: Annotated[
        str,
        typer.Option(
            "--report",
            help="Report format: table (default), json, pareto (Pareto-only rows).",
        ),
    ] = "table",
    verify: Annotated[
        bool,
        typer.Option(
            "--verify",
            help="Verify each envelope's signature before comparing. Exits 1 on failure.",
        ),
    ] = False,
) -> None:
    """Compare two or more benchmark runs and render the Pareto frontier.

    Loads each envelope from a local JSON path, optionally verifies its
    signature, computes Pareto frontiers across the canonical metric pairs
    (quality-vs-cost, throughput-vs-latency, throughput-vs-energy), and
    renders the result as a Rich table (default), JSON, or Pareto-only table.
    """
    if len(run_ids) < 2:
        err_console.print(
            "[red]bench compare requires at least 2 envelope paths.[/red]"
        )
        raise typer.Exit(code=2)

    if report not in {"table", "json", "pareto"}:
        err_console.print(
            f"[red]Unknown --report value:[/red] {report} "
            "(expected one of: table, json, pareto)"
        )
        raise typer.Exit(code=2)

    envelopes: list[tuple[str, Envelope]] = [
        (uri, _load_envelope(uri)) for uri in run_ids
    ]

    if verify:
        for uri, env in envelopes:
            result = verify_envelope(env)
            if not result.ok:
                err_console.print(
                    f"[bold red]FAIL[/bold red]  {uri}: {result.reason}"
                )
                raise typer.Exit(code=1)

    pareto_flags = _compute_all_pareto(envelopes)
    on_any_frontier = _any_frontier(pareto_flags, len(envelopes))

    if report == "json":
        _emit_json(envelopes, pareto_flags)
        return

    only_pareto = report == "pareto"
    _emit_table(envelopes, on_any_frontier, only_pareto=only_pareto)


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _load_envelope(uri: str) -> Envelope:
    """Load an envelope from a URI. Phase 1: local file paths only."""
    if uri.startswith(("hf://", "https://", "s3://")):
        err_console.print(
            f"[red]URI scheme not yet supported in v0.0.0:[/red] "
            f"{uri.split('://')[0]}://"
        )
        err_console.print(
            "Phase 1 supports local file paths only. Download the envelope first."
        )
        raise typer.Exit(code=2)

    path = Path(uri)
    if not path.exists():
        err_console.print(f"[red]Envelope not found:[/red] {path}")
        raise typer.Exit(code=2)

    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        err_console.print(f"[red]Invalid JSON in envelope:[/red] {exc}")
        raise typer.Exit(code=2) from exc

    try:
        return Envelope.model_validate(raw)
    except Exception as exc:
        err_console.print(f"[red]Envelope schema validation failed:[/red] {exc}")
        raise typer.Exit(code=2) from exc


def _metric(env: Envelope, key: str) -> float | None:
    """Pull a metric out of an envelope as a float, ``None`` if absent/null/non-numeric."""
    value = env.metrics.get(key)
    if value is None or isinstance(value, str):
        return None
    return float(value)


def _metric_with_fallback(env: Envelope, *keys: str) -> float | None:
    """Return the first non-None metric across the given keys."""
    for key in keys:
        v = _metric(env, key)
        if v is not None:
            return v
    return None


def _quality_value(env: Envelope) -> float | None:
    """Quality axis: prefer ``goodput_at_slo``, fall back to ``req_per_s_passing``."""
    return _metric_with_fallback(env, "goodput_at_slo", "req_per_s_passing")


def _compute_all_pareto(
    envelopes: list[tuple[str, Envelope]],
) -> dict[str, list[bool]]:
    """Compute Pareto flags for each canonical metric pair across all envelopes."""
    flags: dict[str, list[bool]] = {}
    for label, x_key, y_key, max_x, max_y in _METRIC_PAIRS:
        points: list[tuple[float | None, float | None]] = []
        for _, env in envelopes:
            if label == "quality_vs_cost":
                x_val = _quality_value(env)
            else:
                x_val = _metric(env, x_key)
            y_val = _metric(env, y_key)
            points.append((x_val, y_val))
        flags[label] = _pareto_front(points, maximize_x=max_x, maximize_y=max_y)
    return flags


def _any_frontier(pareto_flags: dict[str, list[bool]], n: int) -> list[bool]:
    """``True`` for indices on the Pareto frontier of any tracked metric pair."""
    result = [False] * n
    for flags in pareto_flags.values():
        for i, on in enumerate(flags):
            if on:
                result[i] = True
    return result


def _fmt(value: float | None) -> str:
    """Format a metric for table display, ``-`` for missing values."""
    if value is None:
        return "-"
    if abs(value) >= 100:
        return f"{value:,.1f}"
    if abs(value) >= 1:
        return f"{value:.2f}"
    return f"{value:.4f}"


def _emit_table(
    envelopes: list[tuple[str, Envelope]],
    on_any_frontier: list[bool],
    *,
    only_pareto: bool,
) -> None:
    """Render the comparison as a Rich table, sorted by throughput desc."""
    rows: list[tuple[int, float, tuple[str, Envelope], bool]] = []
    for idx, (uri, env) in enumerate(envelopes):
        # Sort key: throughput desc. Missing → sort to bottom.
        tput = _metric(env, "throughput_tok_per_s")
        sort_key = -tput if tput is not None else float("inf")
        rows.append((idx, sort_key, (uri, env), on_any_frontier[idx]))

    rows.sort(key=lambda r: r[1])

    table = Table(
        title="Benchmark comparison" + (" (Pareto-only)" if only_pareto else ""),
        show_header=True,
        header_style="bold",
    )
    table.add_column("Suite")
    table.add_column("Model")
    table.add_column("Engine")
    table.add_column("Hardware")
    table.add_column("Throughput tok/s", justify="right")
    table.add_column("TTFT p99 ms", justify="right")
    table.add_column("Goodput @SLO", justify="right")
    table.add_column("$/Mtok", justify="right")
    table.add_column("J/tok", justify="right")
    table.add_column("Pareto?")

    rendered = 0
    for _idx, _key, (_uri, env), on_frontier in rows:
        if only_pareto and not on_frontier:
            continue
        gpu_models = sorted({g.model for g in env.hardware_fingerprint.gpus})
        hw_label = ", ".join(gpu_models) if gpu_models else "-"

        style = "bold" if on_frontier else ""
        table.add_row(
            env.suite_id,
            env.model.id,
            f"{env.engine.name} {env.engine.version}",
            hw_label,
            _fmt(_metric(env, "throughput_tok_per_s")),
            _fmt(_metric(env, "ttft_p99_ms")),
            _fmt(_quality_value(env)),
            _fmt(_metric(env, "cost_usd_per_million_tokens")),
            _fmt(_metric(env, "joules_per_token")),
            "[green]yes[/green]" if on_frontier else "no",
            style=style,
        )
        rendered += 1

    if only_pareto and rendered == 0:
        err_console.print(
            "[yellow]No envelopes on any Pareto frontier "
            "(missing metrics in all inputs?).[/yellow]"
        )

    console.print(table)


def _emit_json(
    envelopes: list[tuple[str, Envelope]],
    pareto_flags: dict[str, list[bool]],
) -> None:
    """Emit the comparison as a single JSON document on stdout."""
    runs: list[dict[str, Any]] = []
    for idx, (uri, env) in enumerate(envelopes):
        runs.append(
            {
                "path": uri,
                "run_id": env.run_id,
                "suite_id": env.suite_id,
                "model_id": env.model.id,
                "engine": {"name": env.engine.name, "version": env.engine.version},
                "metrics": {k: v for k, v in env.metrics.items()},
                "pareto": {
                    label: pareto_flags[label][idx] for label in pareto_flags
                },
            }
        )

    pareto_block: dict[str, list[int]] = {
        label: [i for i, on in enumerate(flags) if on]
        for label, flags in pareto_flags.items()
    }

    payload = {"runs": runs, "pareto": pareto_block}
    console.print_json(data=payload)
