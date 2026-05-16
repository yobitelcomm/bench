"""``bench summary`` — one-glance table of every envelope in a directory.

Loads every ``*.json`` envelope under a directory (or a single file), groups
by ``suite_id``, and renders a Rich table per suite sorted by throughput
descending. Useful after running a sweep that produced many envelopes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from inferencebench.envelope import Envelope

console = Console()
err_console = Console(stderr=True)


# --------------------------------------------------------------------------- #
# Command                                                                     #
# --------------------------------------------------------------------------- #
def summary(
    path: Annotated[
        Path,
        typer.Argument(
            help="Directory to recursively scan for *.json envelopes, or a single envelope file.",
        ),
    ],
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Emit a JSON dict on stdout instead of Rich tables (for piping into jq).",
        ),
    ] = False,
) -> None:
    """Summarise envelopes in a directory or a single file.

    Recursively scans ``path`` for ``*.json`` files (or treats it as a single
    envelope if it points at a file). Envelopes that fail schema validation
    are skipped and reported in the final footer. Output is grouped by
    ``suite_id`` and sorted by ``throughput_tok_per_s`` descending.
    """
    if not path.exists():
        err_console.print(f"[red]Path not found:[/red] {path}")
        raise typer.Exit(code=2)

    candidates = _collect_json_files(path)
    envelopes, skipped = _load_envelopes(candidates)

    suites = _group_by_suite(envelopes)

    if json_output:
        _emit_json(suites, skipped)
        return

    for suite_id in sorted(suites.keys()):
        _emit_suite_table(suite_id, suites[suite_id])

    console.print(
        f"{len(envelopes)} envelopes loaded, "
        f"{skipped} skipped (validation failure), "
        f"{len(suites)} suites"
    )


# --------------------------------------------------------------------------- #
# Loaders                                                                     #
# --------------------------------------------------------------------------- #
def _collect_json_files(path: Path) -> list[Path]:
    """Return every ``*.json`` file under ``path`` (recursive) or just ``path``."""
    if path.is_file():
        return [path]
    return sorted(p for p in path.rglob("*.json") if p.is_file())


def _load_envelopes(paths: list[Path]) -> tuple[list[tuple[Path, Envelope]], int]:
    """Load envelopes; return successful loads plus a skipped count."""
    loaded: list[tuple[Path, Envelope]] = []
    skipped = 0
    for p in paths:
        try:
            raw = json.loads(p.read_text())
            env = Envelope.model_validate(raw)
        except Exception:  # any decode/validate failure → skip
            skipped += 1
            continue
        loaded.append((p, env))
    return loaded, skipped


def _group_by_suite(
    envelopes: list[tuple[Path, Envelope]],
) -> dict[str, list[tuple[Path, Envelope]]]:
    """Bucket envelopes by ``suite_id``."""
    groups: dict[str, list[tuple[Path, Envelope]]] = {}
    for p, env in envelopes:
        groups.setdefault(env.suite_id, []).append((p, env))
    return groups


# --------------------------------------------------------------------------- #
# Field accessors                                                             #
# --------------------------------------------------------------------------- #
def _metric(env: Envelope, key: str) -> float | None:
    """Pull a metric out as a float, ``None`` if absent/null/non-numeric."""
    value = env.metrics.get(key)
    if value is None or isinstance(value, str):
        return None
    return float(value)


def _hardware_label(env: Envelope) -> str:
    """First GPU model from the fingerprint, else ``cpu``."""
    gpus = env.hardware_fingerprint.gpus
    if gpus:
        return gpus[0].model
    return "cpu"


def _quant_label(env: Envelope) -> str:
    """Quantisation format, or ``-`` if not present."""
    if env.quantization is None:
        return "-"
    return env.quantization.format or "-"


def _engine_label(env: Envelope) -> str:
    return f"{env.engine.name} {env.engine.version}"


def _run_id_short(env: Envelope) -> str:
    return env.run_id[:8]


def _fmt(value: float | None) -> str:
    """Format a metric for table display, ``-`` for missing."""
    if value is None:
        return "-"
    if abs(value) >= 100:
        return f"{value:,.1f}"
    if abs(value) >= 1:
        return f"{value:.2f}"
    return f"{value:.4f}"


# --------------------------------------------------------------------------- #
# Table emit                                                                  #
# --------------------------------------------------------------------------- #
def _sort_key(env: Envelope) -> float:
    """Sort by throughput desc — missing throughput sorts to the bottom."""
    tput = _metric(env, "throughput_tok_per_s")
    return -tput if tput is not None else float("inf")


def _emit_suite_table(suite_id: str, rows: list[tuple[Path, Envelope]]) -> None:
    """Render one suite's envelopes as a Rich table."""
    table = Table(
        title=f"Suite: {suite_id}",
        show_header=True,
        header_style="bold",
    )
    table.add_column("Model")
    table.add_column("Engine")
    table.add_column("Quant")
    table.add_column("Hardware")
    table.add_column("Throughput", justify="right")
    table.add_column("TTFT p50/p99", justify="right")
    table.add_column("TPOT p50", justify="right")
    table.add_column("OK rate", justify="right")
    table.add_column("Power avg W", justify="right")
    table.add_column("J/tok", justify="right")
    table.add_column("Cost $/Mtok", justify="right")
    table.add_column("Run ID short")

    sorted_rows = sorted(rows, key=lambda r: _sort_key(r[1]))
    for _path, env in sorted_rows:
        ttft_p50 = _metric(env, "ttft_p50_ms")
        ttft_p99 = _metric(env, "ttft_p99_ms")
        ttft_label = f"{_fmt(ttft_p50)}/{_fmt(ttft_p99)}"
        table.add_row(
            env.model.id,
            _engine_label(env),
            _quant_label(env),
            _hardware_label(env),
            _fmt(_metric(env, "throughput_tok_per_s")),
            ttft_label,
            _fmt(_metric(env, "tpot_p50_ms")),
            _fmt(_metric(env, "ok_rate")),
            _fmt(_metric(env, "power_avg_w")),
            _fmt(_metric(env, "joules_per_token")),
            _fmt(_metric(env, "cost_usd_per_million_tokens")),
            _run_id_short(env),
        )
    console.print(table)


# --------------------------------------------------------------------------- #
# JSON emit                                                                   #
# --------------------------------------------------------------------------- #
def _envelope_summary_dict(env: Envelope) -> dict[str, Any]:
    """Per-envelope summary dict used in ``--json`` output."""
    return {
        "run_id": env.run_id,
        "run_id_short": _run_id_short(env),
        "suite_id": env.suite_id,
        "model_id": env.model.id,
        "engine": {"name": env.engine.name, "version": env.engine.version},
        "quantization": _quant_label(env),
        "hardware": _hardware_label(env),
        "metrics": {
            "throughput_tok_per_s": _metric(env, "throughput_tok_per_s"),
            "ttft_p50_ms": _metric(env, "ttft_p50_ms"),
            "ttft_p99_ms": _metric(env, "ttft_p99_ms"),
            "tpot_p50_ms": _metric(env, "tpot_p50_ms"),
            "ok_rate": _metric(env, "ok_rate"),
            "power_avg_w": _metric(env, "power_avg_w"),
            "joules_per_token": _metric(env, "joules_per_token"),
            "cost_usd_per_million_tokens": _metric(env, "cost_usd_per_million_tokens"),
        },
    }


def _emit_json(
    suites: dict[str, list[tuple[Path, Envelope]]],
    skipped: int,
) -> None:
    """Emit the summary as a single JSON document on stdout."""
    payload: dict[str, Any] = {
        "suites": {
            suite_id: [
                _envelope_summary_dict(env)
                for _p, env in sorted(rows, key=lambda r: _sort_key(r[1]))
            ]
            for suite_id, rows in sorted(suites.items())
        },
        "skipped": skipped,
    }
    console.print_json(data=payload)
