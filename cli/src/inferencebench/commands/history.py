"""``bench history`` — time-series view of one metric across runs.

Track how a specific metric trended across N runs of a stable
benchmark+model combo: e.g. "did our optimisation actually move
throughput forward over the past week?" The command loads every
envelope under a directory, filters by model / suite / engine, sorts
chronologically, and prints a Rich table plus a sparkline of the
chosen metric's evolution.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from inferencebench.envelope import Envelope

console = Console()
err_console = Console(stderr=True)


_SPARK_CHARS = "▁▂▃▄▅▆▇█"


# --------------------------------------------------------------------------- #
# Command                                                                     #
# --------------------------------------------------------------------------- #
def history(
    dir: Annotated[
        Path,
        typer.Argument(
            help="Directory of envelope JSON files (scanned recursively).",
        ),
    ],
    metric: Annotated[
        str,
        typer.Option(
            "--metric",
            help="Metric key to track across runs.",
        ),
    ] = "throughput_tok_per_s",
    filter_model: Annotated[
        str,
        typer.Option(
            "--filter-model",
            help="Only include envelopes whose model.id matches exactly.",
        ),
    ] = "",
    filter_suite: Annotated[
        str,
        typer.Option(
            "--filter-suite",
            help="Only include envelopes whose suite_id matches exactly.",
        ),
    ] = "",
    filter_engine: Annotated[
        str,
        typer.Option(
            "--filter-engine",
            help="Only include envelopes whose engine.name matches exactly.",
        ),
    ] = "",
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Emit a JSON document instead of the Rich table + sparkline.",
        ),
    ] = False,
) -> None:
    """Render the time-series of one metric across a directory of envelopes."""
    if not dir.exists():
        err_console.print(f"[red]Path not found:[/red] {dir}")
        raise typer.Exit(code=2)

    envelopes = _load_envelopes(dir)
    envelopes = _apply_filters(
        envelopes,
        model=filter_model or None,
        suite=filter_suite or None,
        engine=filter_engine or None,
    )
    envelopes.sort(key=lambda e: e.timestamp)

    series = _build_series(envelopes, metric)

    filter_dict: dict[str, str | None] = {
        "model_id": filter_model or None,
        "suite_id": filter_suite or None,
        "engine_name": filter_engine or None,
    }

    if json_output:
        _emit_json(metric, filter_dict, series)
        return

    if not series:
        console.print("[yellow]no matches[/yellow]")
        return

    _emit_table(metric, series)
    _emit_sparkline(series)


# --------------------------------------------------------------------------- #
# Loaders / filters                                                           #
# --------------------------------------------------------------------------- #
def _load_envelopes(root: Path) -> list[Envelope]:
    """Load every parseable envelope under ``root`` (rglob ``*.json``)."""
    out: list[Envelope] = []
    skipped = 0
    for path in sorted(root.rglob("*.json")):
        if not path.is_file():
            continue
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            env = Envelope.model_validate(raw)
        except Exception:  # any decode/validate failure → skip (same policy as `bench summary`)
            skipped += 1
            continue
        out.append(env)
    _ = skipped  # reserved for a future `--show-skipped` flag; not surfaced today
    return out


def _apply_filters(
    envelopes: list[Envelope],
    *,
    model: str | None,
    suite: str | None,
    engine: str | None,
) -> list[Envelope]:
    def keep(env: Envelope) -> bool:
        if model is not None and env.model.id != model:
            return False
        if suite is not None and env.suite_id != suite:
            return False
        if engine is not None and env.engine.name != engine:
            return False
        return True

    return [e for e in envelopes if keep(e)]


# --------------------------------------------------------------------------- #
# Series                                                                      #
# --------------------------------------------------------------------------- #
def _metric_value(env: Envelope, key: str) -> float | None:
    """Pull ``env.metrics[key]`` as float; ``None`` for missing / non-numeric."""
    value = env.metrics.get(key)
    if value is None or isinstance(value, str):
        return None
    return float(value)


def _build_series(envelopes: list[Envelope], metric: str) -> list[dict[str, Any]]:
    """Build the per-envelope datapoint list (chronological, value may be None)."""
    series: list[dict[str, Any]] = []
    for env in envelopes:
        series.append(
            {
                "timestamp": env.timestamp.isoformat(),
                "model_id": env.model.id,
                "suite_id": env.suite_id,
                "engine_name": env.engine.name,
                "engine_version": env.engine.version,
                "run_id": env.run_id,
                "value": _metric_value(env, metric),
            }
        )
    return series


# --------------------------------------------------------------------------- #
# Formatting                                                                  #
# --------------------------------------------------------------------------- #
def _fmt(value: float | None) -> str:
    if value is None:
        return "-"
    if abs(value) >= 100:
        return f"{value:,.1f}"
    if abs(value) >= 1:
        return f"{value:.2f}"
    return f"{value:.4f}"


def _fmt_abs(value: float | None) -> str:
    if value is None:
        return "-"
    sign = "+" if value > 0 else ""
    if abs(value) >= 100:
        return f"{sign}{value:,.1f}"
    if abs(value) >= 1:
        return f"{sign}{value:.2f}"
    return f"{sign}{value:.4f}"


def _fmt_rel(value: float | None) -> str:
    if value is None:
        return "n/a"
    sign = "+" if value > 0 else ""
    return f"{sign}{value * 100:.2f}%"


def _short_ts(iso: str) -> str:
    """Trim the ISO timestamp to ``YYYY-MM-DD HH:MM`` for table display."""
    return iso.replace("T", " ")[:16]


def _trend_cell(curr: float | None, prev: float | None) -> str:
    if curr is None or prev is None:
        return "[dim]-[/dim]"
    if curr > prev:
        return "[green]↑[/green]"
    if curr < prev:
        return "[red]↓[/red]"
    return "[dim]≈[/dim]"


# --------------------------------------------------------------------------- #
# Renderers                                                                   #
# --------------------------------------------------------------------------- #
def _emit_table(metric: str, series: list[dict[str, Any]]) -> None:
    table = Table(
        title=f"History: {metric}",
        show_header=True,
        header_style="bold",
    )
    table.add_column("#", justify="right")
    table.add_column("Timestamp")
    table.add_column("Model")
    table.add_column("Engine")
    table.add_column("Run ID short")
    table.add_column(metric, justify="right")
    table.add_column("Δ vs prev", justify="right")
    table.add_column("Δ vs prev (rel%)", justify="right")
    table.add_column("Trend")

    prev: float | None = None
    single = len(series) == 1
    for i, point in enumerate(series, start=1):
        value = point["value"]
        if single:
            delta_abs_s = "-"
            delta_rel_s = "-"
            trend_s = "[dim]-[/dim]"
        elif i == 1:
            delta_abs_s = "-"
            delta_rel_s = "-"
            trend_s = "[dim]-[/dim]"
        else:
            if value is None or prev is None:
                delta_abs = None
                delta_rel = None
            else:
                delta_abs = value - prev
                delta_rel = (delta_abs / abs(prev)) if prev != 0 else None
            delta_abs_s = _fmt_abs(delta_abs)
            delta_rel_s = _fmt_rel(delta_rel)
            trend_s = _trend_cell(value, prev)

        table.add_row(
            str(i),
            _short_ts(point["timestamp"]),
            point["model_id"],
            f"{point['engine_name']} v{point['engine_version']}",
            point["run_id"][:8],
            _fmt(value),
            delta_abs_s,
            delta_rel_s,
            trend_s,
        )
        prev = value

    console.print(table)


def _spark(values: list[float]) -> str:
    if not values:
        return ""
    lo = min(values)
    hi = max(values)
    if hi == lo:
        # Flat line → mid-block for every point.
        mid = _SPARK_CHARS[len(_SPARK_CHARS) // 2]
        return mid * len(values)
    span = hi - lo
    out: list[str] = []
    last = len(_SPARK_CHARS) - 1
    for v in values:
        # Standard 8-level mapping: scale to [0, 7] inclusive.
        idx = round((v - lo) / span * last)
        idx = max(0, min(last, idx))
        out.append(_SPARK_CHARS[idx])
    return "".join(out)


def _emit_sparkline(series: list[dict[str, Any]]) -> None:
    values = [p["value"] for p in series if p["value"] is not None]
    if len(values) < 2:
        return
    bar = _spark(values)
    mn = min(values)
    mx = max(values)
    med = statistics.median(values)
    console.print(
        f"{bar}  min={_fmt(mn)}  median={_fmt(med)}  max={_fmt(mx)}"
    )


# --------------------------------------------------------------------------- #
# JSON                                                                        #
# --------------------------------------------------------------------------- #
def _emit_json(
    metric: str,
    filter_dict: dict[str, str | None],
    series: list[dict[str, Any]],
) -> None:
    values = [p["value"] for p in series if p["value"] is not None]
    stats: dict[str, float | None]
    if values:
        stats = {
            "min": min(values),
            "max": max(values),
            "median": statistics.median(values),
            "first": values[0],
            "last": values[-1],
        }
    else:
        stats = {"min": None, "max": None, "median": None, "first": None, "last": None}

    payload: dict[str, Any] = {
        "metric": metric,
        "filter": filter_dict,
        "series": [
            {
                "timestamp": p["timestamp"],
                "model_id": p["model_id"],
                "value": p["value"],
                "run_id": p["run_id"],
            }
            for p in series
        ],
        "stats": stats,
    }
    console.print_json(data=payload)
