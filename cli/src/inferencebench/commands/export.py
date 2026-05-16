"""``bench export`` — convert an envelope into share-friendly text formats.

Envelopes are dense JSON blobs. ``bench export`` collapses them into
copy-paste-friendly markdown, CSV, or Slack/Discord snippets so results can
land in a PR comment, a spreadsheet, or a chat channel without bespoke
formatting steps.

Phase 1 supports local envelope paths only; remote URIs go through
``bench fetch`` first.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from inferencebench.envelope import Envelope

console = Console()
err_console = Console(stderr=True)


# --------------------------------------------------------------------------- #
# Supported formats                                                           #
# --------------------------------------------------------------------------- #
_FORMATS: frozenset[str] = frozenset({"markdown", "csv", "slack"})


# --------------------------------------------------------------------------- #
# Command                                                                     #
# --------------------------------------------------------------------------- #
def export(
    envelope_path: Annotated[
        str,
        typer.Argument(help="Path to the envelope JSON file (local file only)."),
    ],
    format: Annotated[
        str,
        typer.Option(
            "--format",
            help="Output format: markdown (default), csv, or slack.",
        ),
    ] = "markdown",
    out: Annotated[
        Path | None,
        typer.Option(
            "--out",
            help="Write to this file instead of stdout.",
        ),
    ] = None,
    metric: Annotated[
        list[str] | None,
        typer.Option(
            "--metric",
            help=(
                "Only include these metric keys. Repeatable. Default: include "
                "every metric present in the envelope."
            ),
        ),
    ] = None,
) -> None:
    """Render an envelope as markdown, CSV, or a Slack snippet.

    Loads ``envelope_path`` (local file), filters metrics if ``--metric`` is
    given, and writes the result to ``--out`` or stdout.
    """
    if format not in _FORMATS:
        err_console.print(
            f"[red]Unknown --format value:[/red] {format} "
            f"(expected one of: markdown, csv, slack)"
        )
        raise typer.Exit(code=2)

    envelope = _load_envelope(envelope_path)
    filtered_metrics = _filter_metrics(envelope.metrics, metric)

    if format == "markdown":
        rendered = _render_markdown(envelope, filtered_metrics)
    elif format == "csv":
        rendered = _render_csv(envelope, filtered_metrics)
    else:  # slack
        rendered = _render_slack(envelope, filtered_metrics)

    if out is not None:
        out.write_text(rendered, encoding="utf-8")
    else:
        # ``print`` (not console.print) keeps the output verbatim — Rich would
        # otherwise interpret square brackets in markdown table headers as
        # markup tags.
        print(rendered, end="" if rendered.endswith("\n") else "\n")


# --------------------------------------------------------------------------- #
# Loading                                                                     #
# --------------------------------------------------------------------------- #
def _load_envelope(uri: str) -> Envelope:
    """Load an envelope from a local file path."""
    if uri.startswith(("hf://", "https://", "s3://")):
        err_console.print(
            f"[red]URI scheme not yet supported in v0.0.0:[/red] "
            f"{uri.split('://')[0]}://"
        )
        err_console.print(
            "Use `bench fetch` to download the envelope first, then export the local file."
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
# Metric helpers                                                              #
# --------------------------------------------------------------------------- #
def _filter_metrics(
    metrics: dict[str, float | int | str | None],
    keep: list[str] | None,
) -> dict[str, float | int | str | None]:
    """Drop ``None`` values and optionally restrict to ``keep``.

    Default (``keep is None``) keeps every metric the envelope carries that
    isn't null. ``--metric foo --metric bar`` keeps only those keys (and only
    when they're present + non-null in the envelope).
    """
    if keep is None:
        return {k: v for k, v in metrics.items() if v is not None}
    wanted = set(keep)
    return {k: v for k, v in metrics.items() if k in wanted and v is not None}


def _fmt_metric(value: float | int | str | None) -> str:
    """Format a metric value for human-readable display.

    Numeric values use ``.4g`` (compact significant-digit formatting).
    Strings render verbatim. ``None`` becomes ``-`` though callers usually
    drop ``None`` first.
    """
    if value is None:
        return "-"
    if isinstance(value, str):
        return value
    return f"{float(value):.4g}"


def _hardware_label(envelope: Envelope) -> tuple[str, int]:
    """Return ``(first GPU model | "cpu", num GPUs)``."""
    gpus = envelope.hardware_fingerprint.gpus
    if not gpus:
        return ("cpu", 0)
    return (gpus[0].model, len(gpus))


def _signing_label(envelope: Envelope) -> str:
    """Human-readable signing method, or ``unsigned``."""
    if envelope.signature is None:
        return "unsigned"
    return envelope.signature.method


# --------------------------------------------------------------------------- #
# Markdown                                                                    #
# --------------------------------------------------------------------------- #
def _render_markdown(
    envelope: Envelope,
    metrics: dict[str, float | int | str | None],
) -> str:
    """Render a markdown header + metric table."""
    gpu_model, num_gpus = _hardware_label(envelope)
    dataset_hash_short = envelope.dataset.hash[:12]
    signing = _signing_label(envelope)
    verify_hint = f"bench verify {envelope.content_hash()[:12]}.json"

    lines: list[str] = []
    lines.append(f"## InferenceBench result — `{envelope.suite_id}`")
    lines.append("")
    lines.append(
        f"- **Model**: `{envelope.model.id}` (revision `{envelope.model.revision}`)"
    )
    lines.append(
        f"- **Engine**: `{envelope.engine.name} v{envelope.engine.version}`"
    )
    lines.append(f"- **Hardware**: `{gpu_model}` x `{num_gpus}`")
    lines.append(
        f"- **Dataset**: `{envelope.dataset.id}` (sha256 `{dataset_hash_short}`)"
    )
    lines.append(f"- **Run ID**: `{envelope.run_id}`")
    lines.append(f"- **Signed**: `{signing}` — verify with `{verify_hint}`")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---|")
    for key in sorted(metrics.keys()):
        lines.append(f"| {key} | {_fmt_metric(metrics[key])} |")
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CSV                                                                         #
# --------------------------------------------------------------------------- #
def _render_csv(
    envelope: Envelope,
    metrics: dict[str, float | int | str | None],
) -> str:
    """Render a metric,value CSV prefixed with ``#`` comment header rows."""
    gpu_model, num_gpus = _hardware_label(envelope)
    lines: list[str] = []
    lines.append(f"# suite_id={envelope.suite_id}")
    lines.append(f"# suite_version={envelope.suite_version}")
    lines.append(f"# run_id={envelope.run_id}")
    lines.append(f"# model_id={envelope.model.id}")
    lines.append(f"# model_revision={envelope.model.revision}")
    lines.append(f"# engine={envelope.engine.name} v{envelope.engine.version}")
    lines.append(f"# hardware={gpu_model} x{num_gpus}")
    lines.append(f"# dataset_id={envelope.dataset.id}")
    lines.append(f"# content_hash={envelope.content_hash()}")
    lines.append(f"# signed={_signing_label(envelope)}")
    lines.append("metric,value")
    for key in sorted(metrics.keys()):
        lines.append(f"{key},{_csv_escape(metrics[key])}")
    return "\n".join(lines) + "\n"


def _csv_escape(value: float | int | str | None) -> str:
    """Minimal CSV escaping. Quote strings containing commas / quotes / newlines."""
    if value is None:
        return ""
    if isinstance(value, str):
        if any(c in value for c in (",", "\"", "\n", "\r")):
            escaped = value.replace("\"", "\"\"")
            return f'"{escaped}"'
        return value
    return f"{float(value):.4g}"


# --------------------------------------------------------------------------- #
# Slack                                                                       #
# --------------------------------------------------------------------------- #
# Metrics handled by the curated slack block. Anything else falls through to
# a generic ``  key: value`` line so plugins that emit custom metrics still
# show up.
_SLACK_CURATED: frozenset[str] = frozenset(
    {
        "throughput_tok_per_s",
        "ttft_p50_ms",
        "ttft_p99_ms",
        "tpot_p50_ms",
        "tpot_p99_ms",
        "total_p50_ms",
        "total_p99_ms",
        "ok_rate",
        "compliance_rate",
        "power_avg_w",
        "power_peak_w",
        "joules_per_token",
    }
)


def _render_slack(
    envelope: Envelope,
    metrics: dict[str, float | int | str | None],
) -> str:
    """Render a compact, Slack/Discord-friendly fenced code block."""
    gpu_model, num_gpus = _hardware_label(envelope)
    lines: list[str] = []
    lines.append("```")
    lines.append("\U0001f680 InferenceBench result")
    lines.append(f"suite: {envelope.suite_id}")
    lines.append(f"model: {envelope.model.id}")
    lines.append(f"engine: {envelope.engine.name} v{envelope.engine.version}")
    lines.append(f"hardware: {gpu_model} x {num_gpus}")
    lines.append("metrics:")

    metric_lines = _slack_metric_lines(metrics, slo_template=envelope.slo_template)
    lines.extend(metric_lines)

    lines.append(f"verify: bench verify {envelope.content_hash()[:12]}.json")
    lines.append("```")
    return "\n".join(lines) + "\n"


class _MetricTake:
    """Tracks which metric keys have been consumed by curated formatters."""

    def __init__(self, metrics: dict[str, float | int | str | None]) -> None:
        self._metrics = metrics
        self.consumed: set[str] = set()

    def take(self, key: str) -> float | int | str | None:
        if key in self._metrics:
            self.consumed.add(key)
            return self._metrics[key]
        return None


def _latency_pair_line(
    take: _MetricTake, base: str
) -> str | None:
    """Render ``ttft`` / ``tpot`` / ``total`` as a p50 (p99 …) pair if available."""
    p50 = take.take(f"{base}_p50_ms")
    p99 = take.take(f"{base}_p99_ms")
    if p50 is not None and p99 is not None:
        return f"  {base}_p50_ms: {_fmt_metric(p50)} (p99 {_fmt_metric(p99)})"
    if p50 is not None:
        return f"  {base}_p50_ms: {_fmt_metric(p50)}"
    if p99 is not None:
        return f"  {base}_p99_ms: {_fmt_metric(p99)}"
    return None


def _power_line(take: _MetricTake) -> str | None:
    """Render the ``power: X W avg, Y W peak`` summary line if values exist."""
    p_avg = take.take("power_avg_w")
    p_peak = take.take("power_peak_w")
    if p_avg is not None and p_peak is not None:
        return (
            f"  power: {_fmt_metric(p_avg)} W avg, "
            f"{_fmt_metric(p_peak)} W peak"
        )
    if p_avg is not None:
        return f"  power_avg_w: {_fmt_metric(p_avg)}"
    if p_peak is not None:
        return f"  power_peak_w: {_fmt_metric(p_peak)}"
    return None


def _slack_metric_lines(
    metrics: dict[str, float | int | str | None],
    *,
    slo_template: str,
) -> list[str]:
    """Build the indented metric body of the Slack block.

    Pairs p50/p99 latencies onto one line when both are present, renders
    ok_rate / compliance_rate as percentages, and groups power + energy
    when both halves are available.
    """
    take = _MetricTake(metrics)
    out: list[str] = []

    tput = take.take("throughput_tok_per_s")
    if tput is not None:
        out.append(f"  throughput_tok_per_s: {_fmt_metric(tput)}")

    for base in ("ttft", "tpot", "total"):
        line = _latency_pair_line(take, base)
        if line is not None:
            out.append(line)

    ok_rate = take.take("ok_rate")
    if ok_rate is not None:
        out.append(f"  ok_rate: {_pct(ok_rate)}")

    compliance = take.take("compliance_rate")
    if compliance is not None:
        slo_suffix = f" @ {slo_template}" if slo_template else ""
        out.append(f"  compliance: {_pct(compliance)}{slo_suffix}")

    power = _power_line(take)
    if power is not None:
        out.append(power)

    j_per_tok = take.take("joules_per_token")
    if j_per_tok is not None:
        out.append(f"  energy: {_fmt_metric(j_per_tok)} J/tok")

    # Anything not in the curated set falls through verbatim so unusual
    # plugin-specific metrics still surface.
    for key in sorted(metrics.keys()):
        if key in take.consumed or key in _SLACK_CURATED:
            continue
        out.append(f"  {key}: {_fmt_metric(metrics[key])}")

    if not out:
        out.append("  (no metrics)")
    return out


def _pct(value: float | int | str | None) -> str:
    """Render a 0..1 rate as a percentage. Strings/None passed through verbatim."""
    if value is None:
        return "-"
    if isinstance(value, str):
        return value
    v = float(value)
    # Some plugins emit rates already in % (0..100), some in fractions (0..1).
    # We assume the canonical envelope convention is 0..1; anything > 1 is
    # treated as already-percent.
    if v <= 1.0:
        v *= 100.0
    # Drop the trailing .0 on whole-number percents (100% not 100.0%).
    if v == int(v):
        return f"{int(v)}%"
    return f"{v:.1f}%"
