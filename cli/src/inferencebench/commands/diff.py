"""``bench diff`` — per-metric delta between two envelopes.

Sharper than ``bench compare`` (which renders Pareto frontiers across many
runs): given exactly two envelopes — a baseline and a candidate — emit every
metric's absolute + relative delta and classify it as
improvement / regression / no-change. The canonical "did my optimisation
actually help?" command.

Direction policy is per-metric (lower-is-better for latencies / cost /
energy; higher-is-better for throughput / quality / goodput). Metrics with
no known direction are tagged ``unknown`` — delta still shown, no verdict.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Annotated, Any, Literal

import typer
from rich.console import Console
from rich.table import Table

from inferencebench.envelope import Envelope, verify_envelope

console = Console()
err_console = Console(stderr=True)


# --------------------------------------------------------------------------- #
# Direction policy                                                            #
# --------------------------------------------------------------------------- #
# Metrics where a smaller value is better (latencies, cost, energy, power,
# error rates — WER/CER for voice transcription).
_LOWER_IS_BETTER: frozenset[str] = frozenset(
    {
        "ttft_p50_ms",
        "ttft_p99_ms",
        "tpot_p50_ms",
        "tpot_p99_ms",
        "total_p50_ms",
        "total_p99_ms",
        "joules_per_token",
        "cost_usd_per_million_tokens",
        "power_avg_w",
        "power_peak_w",
        "energy_joules_total",
        # voice.transcription error rates
        "wer_mean",
        "wer_p50",
        "wer_p95",
        "cer_mean",
        "cer_p50",
        "cer_p95",
        # code.generation execution-timeout rate
        "timeout_rate",
    }
)

# Metrics where a larger value is better (throughput, quality, goodput,
# retrieval hit metrics — recall/MRR/nDCG).
_HIGHER_IS_BETTER: frozenset[str] = frozenset(
    {
        "throughput_tok_per_s",
        "req_per_s_passing",
        "req_per_s_all",
        "compliance_rate",
        "ok_rate",
        "goodput_at_slo",
        "accuracy",
        "accuracy_p05",
        "accuracy_p50",
        "accuracy_p95",
        # embeddings.retrieval hit metrics
        "recall_at_5_mean",
        "recall_at_5_p50",
        "mrr_at_10_mean",
        "mrr_at_10_p50",
        "ndcg_at_10_mean",
        "ndcg_at_10_p50",
        # llm.mt translation quality (chrF / BLEU / exact-match)
        "chrf_mean",
        "chrf_p50",
        "chrf_p95",
        "bleu_mean",
        "bleu_p50",
        "bleu_p95",
        "exact_match_rate",
        # code.generation pass@k metrics
        "pass_at_1",
        "pass_at_1_p05",
        "pass_at_1_p50",
        "pass_at_1_p95",
        "pass_at_k",
        "pass_at_k_mean",
    }
)


Verdict = Literal["improvement", "regression", "no_change", "unknown", "missing"]


# --------------------------------------------------------------------------- #
# Command                                                                     #
# --------------------------------------------------------------------------- #
def diff(
    baseline_path: Annotated[
        str,
        typer.Argument(help="Path to the baseline envelope (local file)."),
    ],
    candidate_path: Annotated[
        str,
        typer.Argument(help="Path to the candidate envelope (local file)."),
    ],
    tolerance: Annotated[
        float,
        typer.Option(
            "--tolerance",
            help=(
                "Relative-delta band (default 0.02 = 2%) within which a metric "
                "is classified as 'no_change'."
            ),
            min=0.0,
        ),
    ] = 0.02,
    report: Annotated[
        str,
        typer.Option("--report", help="Report format: table (default) or json."),
    ] = "table",
    verify: Annotated[
        bool,
        typer.Option(
            "--verify",
            help="Verify both envelopes' signatures before diffing. Exits 1 on failure.",
        ),
    ] = False,
    strict: Annotated[
        bool,
        typer.Option(
            "--strict",
            help="Exit 1 if any metric is classified as a regression.",
        ),
    ] = False,
) -> None:
    """Diff two envelopes and report per-metric improvement / regression.

    Loads ``baseline_path`` and ``candidate_path`` (local envelope JSON
    files), optionally verifies them, then renders every metric's absolute
    and relative delta along with a direction-aware verdict.
    """
    if report not in {"table", "json"}:
        err_console.print(
            f"[red]Unknown --report value:[/red] {report} "
            "(expected one of: table, json)"
        )
        raise typer.Exit(code=2)

    baseline = _load_envelope(baseline_path)
    candidate = _load_envelope(candidate_path)

    if verify:
        for uri, env in ((baseline_path, baseline), (candidate_path, candidate)):
            result = verify_envelope(env)
            if not result.ok:
                err_console.print(
                    f"[bold red]FAIL[/bold red]  {uri}: {result.reason}"
                )
                raise typer.Exit(code=1)

    context_match = _context_match(baseline, candidate)
    rows = _compute_rows(baseline, candidate, tolerance=tolerance)
    has_regression = any(r["verdict"] == "regression" for r in rows)

    if report == "json":
        _emit_json(baseline_path, candidate_path, context_match, rows)
    else:
        _emit_table(context_match, rows)

    if strict and has_regression:
        raise typer.Exit(code=1)
    raise typer.Exit(code=0)


# --------------------------------------------------------------------------- #
# Loading                                                                     #
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


# --------------------------------------------------------------------------- #
# Diff core                                                                   #
# --------------------------------------------------------------------------- #
def _direction(metric: str) -> Literal["lower", "higher", "unknown"]:
    if metric in _LOWER_IS_BETTER:
        return "lower"
    if metric in _HIGHER_IS_BETTER:
        return "higher"
    return "unknown"


def _as_float(value: float | int | str | None) -> float | None:
    """Coerce a metric to ``float`` or ``None`` for null / NaN / non-numeric values.

    String-valued metrics (e.g. ``cost_source = "registry:groq"``) collapse to
    ``None`` here so the numeric diff machinery skips them gracefully — the
    row still appears in the table but is classified as ``unknown`` / no-delta.
    """
    if value is None:
        return None
    if isinstance(value, str):
        return None
    f = float(value)
    if math.isnan(f):
        return None
    return f


def _classify(
    metric: str,
    baseline: float | None,
    candidate: float | None,
    *,
    tolerance: float,
) -> tuple[float | None, float | None, Verdict]:
    """Return ``(delta_abs, delta_rel, verdict)`` for one metric pair."""
    # Missing in candidate but present in baseline → "missing".
    if baseline is not None and candidate is None:
        return None, None, "missing"
    # Present in candidate but missing in baseline → can't compute delta,
    # but it's not a regression — leave as unknown (special "new metric").
    if baseline is None and candidate is not None:
        return None, None, "unknown"
    # Both missing / both NaN → no_change.
    if baseline is None and candidate is None:
        return None, None, "no_change"

    assert baseline is not None
    assert candidate is not None
    delta_abs = candidate - baseline
    if baseline == 0:
        delta_rel: float | None = None
    else:
        delta_rel = delta_abs / abs(baseline)

    direction = _direction(metric)
    # If we can't compute a relative delta, still try to classify against the
    # absolute delta when direction is known; otherwise mark unknown.
    if direction == "unknown":
        return delta_abs, delta_rel, "unknown"

    # Tolerance band uses the relative delta when available, absolute delta
    # otherwise (baseline == 0 case).
    if delta_rel is not None:
        if abs(delta_rel) <= tolerance:
            return delta_abs, delta_rel, "no_change"
    elif delta_abs == 0:
        return delta_abs, delta_rel, "no_change"

    candidate_better = (
        (delta_abs < 0) if direction == "lower" else (delta_abs > 0)
    )
    verdict: Verdict = "improvement" if candidate_better else "regression"
    return delta_abs, delta_rel, verdict


def _compute_rows(
    baseline: Envelope,
    candidate: Envelope,
    *,
    tolerance: float,
) -> list[dict[str, Any]]:
    """Build one row per metric (union of baseline + candidate keys)."""
    names = sorted(set(baseline.metrics.keys()) | set(candidate.metrics.keys()))
    rows: list[dict[str, Any]] = []
    for name in names:
        b = _as_float(baseline.metrics.get(name))
        c = _as_float(candidate.metrics.get(name))
        delta_abs, delta_rel, verdict = _classify(
            name, b, c, tolerance=tolerance
        )
        rows.append(
            {
                "name": name,
                "baseline": b,
                "candidate": c,
                "delta_abs": delta_abs,
                "delta_rel": delta_rel,
                "verdict": verdict,
                "direction": _direction(name),
            }
        )
    return rows


# --------------------------------------------------------------------------- #
# Context match                                                               #
# --------------------------------------------------------------------------- #
def _context_match(baseline: Envelope, candidate: Envelope) -> dict[str, Any]:
    """Compare cross-cutting envelope identity fields. Used for warnings."""

    def _quant(env: Envelope) -> str | None:
        return env.quantization.format if env.quantization is not None else None

    fields = {
        "suite_id": (baseline.suite_id, candidate.suite_id),
        "model_id": (baseline.model.id, candidate.model.id),
        "engine_name": (baseline.engine.name, candidate.engine.name),
        "engine_version": (baseline.engine.version, candidate.engine.version),
        "quantization_format": (_quant(baseline), _quant(candidate)),
        "hardware_fingerprint": (
            baseline.hardware_fingerprint.fingerprint_sha256,
            candidate.hardware_fingerprint.fingerprint_sha256,
        ),
    }
    matches: dict[str, Any] = {}
    for field, (b, c) in fields.items():
        matches[field] = {
            "baseline": b,
            "candidate": c,
            "match": b == c,
        }
    matches["all_match"] = all(v["match"] for v in matches.values() if isinstance(v, dict))
    return matches


# --------------------------------------------------------------------------- #
# Rendering                                                                   #
# --------------------------------------------------------------------------- #
def _fmt_value(value: float | None) -> str:
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


def _verdict_cell(row: dict[str, Any]) -> str:
    verdict = row["verdict"]
    direction = row["direction"]
    delta_abs = row["delta_abs"]
    if verdict == "no_change":
        return "[dim]≈[/dim]"
    if verdict == "unknown":
        return "[dim]?[/dim]"
    if verdict == "missing":
        return "[red]missing[/red]"
    # improvement / regression need an arrow showing which way the metric
    # moved, coloured by whether that move is good or bad.
    arrow = "↑" if delta_abs is not None and delta_abs > 0 else "↓"
    colour = "green" if verdict == "improvement" else "red"
    label = "improvement" if verdict == "improvement" else "regression"
    _ = direction  # arrow direction is independent of which-is-better policy
    return f"[{colour}]{arrow} {label}[/{colour}]"


def _sort_rows_for_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort rows for table display.

    Regressions (worst rel-delta first), then improvements (best first),
    then no_change / unknown / missing alphabetically at the bottom.
    """

    def bucket(row: dict[str, Any]) -> int:
        v = row["verdict"]
        if v == "regression":
            return 0
        if v == "improvement":
            return 1
        return 2

    def severity(row: dict[str, Any]) -> float:
        rel = row["delta_rel"]
        if rel is None:
            return 0.0
        direction = row["direction"]
        # For lower-is-better metrics, positive rel = worse; flip sign so
        # "worst" is always the largest positive number.
        sign = 1.0 if direction == "lower" else -1.0
        return float(sign * rel)

    def key(row: dict[str, Any]) -> tuple[int, float, str]:
        b = bucket(row)
        if b == 0:  # regression: worst first → largest severity first
            return (b, -severity(row), row["name"])
        if b == 1:  # improvement: best first → largest |delta| first
            return (b, severity(row), row["name"])
        return (b, 0.0, row["name"])

    return sorted(rows, key=key)


def _emit_table(
    context_match: dict[str, Any],
    rows: list[dict[str, Any]],
) -> None:
    """Render the diff as a Rich table, prefixed with any context warning."""
    if not context_match["all_match"]:
        diffs = [
            f"{k} ({v['baseline']} -> {v['candidate']})"
            for k, v in context_match.items()
            if isinstance(v, dict) and not v["match"]
        ]
        console.print(
            "[yellow]warning: baseline and candidate differ on: "
            + ", ".join(diffs)
            + "[/yellow]"
        )
        console.print(
            "[yellow]diffing across different contexts is supported, but "
            "interpret the deltas with care.[/yellow]"
        )

    table = Table(
        title="Envelope diff",
        show_header=True,
        header_style="bold",
    )
    table.add_column("Metric")
    table.add_column("Baseline", justify="right")
    table.add_column("Candidate", justify="right")
    table.add_column("Δ abs", justify="right")
    table.add_column("Δ rel", justify="right")
    table.add_column("Verdict")

    for row in _sort_rows_for_table(rows):
        table.add_row(
            row["name"],
            _fmt_value(row["baseline"]),
            _fmt_value(row["candidate"]),
            _fmt_abs(row["delta_abs"]),
            _fmt_rel(row["delta_rel"]),
            _verdict_cell(row),
        )

    console.print(table)


def _emit_json(
    baseline_path: str,
    candidate_path: str,
    context_match: dict[str, Any],
    rows: list[dict[str, Any]],
) -> None:
    """Emit the diff as a single JSON document on stdout."""
    payload = {
        "baseline_path": baseline_path,
        "candidate_path": candidate_path,
        "context_match": context_match,
        "metrics": [
            {
                "name": r["name"],
                "baseline": r["baseline"],
                "candidate": r["candidate"],
                "delta_abs": r["delta_abs"],
                "delta_rel": r["delta_rel"],
                "verdict": r["verdict"],
            }
            for r in rows
        ],
    }
    console.print_json(data=payload)
