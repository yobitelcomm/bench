"""``bench replay`` — reproduce a benchmark run from an existing envelope.

Reproducibility is the product's moat. The envelope already records every
input needed to re-run a benchmark (``suite_id``, ``model.id``, ``engine.name``,
``dataset.id``, ``seed``, ``slo_template``, quantization). What it deliberately
does NOT record is the live engine endpoint — that is host-specific and would
make envelopes non-portable. ``bench replay`` consumes an envelope plus a fresh
``--base-url`` and produces a NEW envelope that can be diffed against the
original.

Plugin discovery + RunContext construction reuses the helpers in
:mod:`inferencebench.commands.run` so we never go out of sync with how a
fresh run is wired up.
"""

from __future__ import annotations

import json
import traceback
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.status import Status
from rich.table import Table

from inferencebench.commands.run import (
    _build_signing_extra,
    _entry_points,
    _find_entry_point,
    _resolve_plugin_schemas,
    _split_suite_id,
    _write_envelope,
)
from inferencebench.envelope import Envelope, verify_envelope

console = Console()
err_console = Console(stderr=True)


# --------------------------------------------------------------------------- #
# Envelope loader                                                             #
# --------------------------------------------------------------------------- #
def _load_envelope(uri: str) -> Envelope:
    """Load an envelope from a local path. Phase 1: local paths only."""
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
# Summary table                                                               #
# --------------------------------------------------------------------------- #
_HEADLINE_METRICS: tuple[str, ...] = (
    "throughput_tok_per_s",
    "ttft_p50_ms",
    "ttft_p99_ms",
    "tpot_p50_ms",
    "total_p50_ms",
    "ok_rate",
    "compliance_rate",
    "req_per_s_passing",
    "cost_usd_per_million_tokens",
    "power_avg_w",
    "joules_per_token",
)


def _fmt_metric(value: Any) -> str:  # noqa: ANN401
    if isinstance(value, int | float):
        return f"{value:.4g}"
    return "-"


def _print_replay_summary(
    source: Envelope,
    source_uri: str,
    replay: Envelope,
    replay_path: Path,
) -> None:
    """Render a side-by-side Rich summary so the user can eyeball reproducibility."""
    header = Table(title="Replay summary", show_header=True, header_style="bold")
    header.add_column("field", style="cyan")
    header.add_column("source", style="bold")
    header.add_column("replay", style="bold")
    header.add_column("match", justify="center")

    def _row(label: str, a: str, b: str) -> None:
        ok = a == b
        marker = "[green]yes[/green]" if ok else "[red]no[/red]"
        header.add_row(label, a, b, marker)

    _row("envelope", source_uri, str(replay_path))
    _row("suite_id", source.suite_id, replay.suite_id)
    _row("model.id", source.model.id, replay.model.id)
    _row(
        "engine.name",
        source.engine.name,
        replay.engine.name,
    )
    src_quant = source.quantization.format if source.quantization else ""
    rep_quant = replay.quantization.format if replay.quantization else ""
    _row("quantization", src_quant or "-", rep_quant or "-")
    _row("dataset.id", source.dataset.id, replay.dataset.id)
    _row("seed", str(source.seed), str(replay.seed))
    _row("slo_template", source.slo_template, replay.slo_template)
    console.print(header)

    metrics_table = Table(
        title="Headline metrics (source vs replay)",
        show_header=True,
        header_style="bold",
    )
    metrics_table.add_column("metric", style="cyan")
    metrics_table.add_column("source", justify="right")
    metrics_table.add_column("replay", justify="right")

    keys: list[str] = []
    for key in _HEADLINE_METRICS:
        if key in source.metrics or key in replay.metrics:
            keys.append(key)
    for key in keys:
        metrics_table.add_row(
            key,
            _fmt_metric(source.metrics.get(key)),
            _fmt_metric(replay.metrics.get(key)),
        )
    console.print(metrics_table)
    console.print(
        "[dim]Use [bold]bench compare[/bold] for the full Pareto-aware diff.[/dim]"
    )


# --------------------------------------------------------------------------- #
# CLI command                                                                 #
# --------------------------------------------------------------------------- #
def replay(
    envelope_path: Annotated[
        str,
        typer.Argument(help="Path to the source envelope JSON to replay."),
    ],
    base_url: Annotated[
        str,
        typer.Option(
            "--base-url",
            help=(
                "Engine base URL for the replay (e.g. http://localhost:8000/v1). "
                "Required: envelopes are deliberately host-agnostic and do not "
                "store live endpoints."
            ),
        ),
    ] = "",
    output: Annotated[
        str,
        typer.Option(
            "--output",
            help="Output directory for the replay envelope.",
        ),
    ] = "",
    signing_mode: Annotated[
        str,
        typer.Option(
            "--signing-mode",
            help="Envelope signing mode: 'dev' (local cosign key) or 'keyless'.",
        ),
    ] = "dev",
    dev_key: Annotated[
        str,
        typer.Option(
            "--dev-key",
            help="Path to local cosign signing key (used when --signing-mode=dev).",
        ),
    ] = "",
    verify: Annotated[
        bool,
        typer.Option(
            "--verify/--no-verify",
            help=(
                "Verify the source envelope's signature before replaying. "
                "A bad envelope shouldn't seed a replay."
            ),
        ),
    ] = True,
) -> None:
    """Reproduce a benchmark run from an existing envelope.

    Loads the source envelope, (optionally) verifies its signature, then
    re-runs the exact same suite/model/engine/dataset/seed configuration
    against a fresh engine endpoint supplied via ``--base-url``. Writes a
    new signed envelope to ``--output`` so the two can be diffed.
    """
    source = _load_envelope(envelope_path)

    if verify:
        result = verify_envelope(source)
        if not result.ok:
            err_console.print(
                f"[bold red]FAIL[/bold red] source envelope failed verification: "
                f"{result.reason}"
            )
            err_console.print(
                "[red]Refusing to replay an unverified envelope.[/red] "
                "Pass [bold]--no-verify[/bold] to bypass (e.g. for unsigned local fixtures)."
            )
            raise typer.Exit(code=1)

    if not base_url:
        err_console.print(
            "[red]--base-url is required for bench replay.[/red] "
            "Envelopes are host-agnostic and do not store the live engine URL — "
            "you must point this replay at a running engine "
            "(e.g. [bold]--base-url http://localhost:8000/v1[/bold])."
        )
        raise typer.Exit(code=2)

    # Plugin discovery via the envelope's suite_id.
    eps = _entry_points()
    plugin_name, _full_id = _split_suite_id(source.suite_id)
    ep = _find_entry_point(eps, plugin_name, source.suite_id)

    try:
        plugin_cls = ep.load()
    except Exception as exc:  # pragma: no cover - defensive
        err_console.print(f"[red]Failed to load plugin '{ep.name}':[/red] {exc}")
        raise typer.Exit(code=1) from exc
    plugin = plugin_cls()

    try:
        spec = plugin.get_benchmark(source.suite_id)
    except KeyError:
        err_console.print(
            f"[red]Plugin '{ep.name}' no longer ships benchmark id "
            f"'{source.suite_id}'.[/red] "
            "The plugin may have been upgraded and dropped this benchmark. "
            "Pin to the plugin version recorded in the source envelope to replay."
        )
        raise typer.Exit(code=1) from None
    except Exception as exc:  # pragma: no cover - defensive
        err_console.print(f"[red]Plugin error while resolving spec:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    try:
        run_context_cls, engine_kind_cls = _resolve_plugin_schemas(ep)
    except RuntimeError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    try:
        engine_kind = engine_kind_cls(source.engine.name)
    except ValueError as exc:
        err_console.print(
            f"[red]Unknown engine in source envelope:[/red] {source.engine.name}"
        )
        raise typer.Exit(code=1) from exc

    output_dir = Path(output) if output else Path.cwd() / "replay-results"
    try:
        signing_extra = _build_signing_extra(signing_mode, dev_key)
    except typer.Exit:
        raise

    quant_fmt = source.quantization.format if source.quantization else ""

    try:
        ctx = run_context_cls(
            model_id=source.model.id,
            engine_kind=engine_kind,
            base_url=base_url,
            quantization_format=quant_fmt,
            output_dir=output_dir,
            extra=dict(signing_extra),
        )
    except Exception as exc:
        err_console.print(f"[red]Invalid run context:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    try:
        with Status("[bold]Replaying benchmark…[/bold]", console=err_console):
            new_envelope = plugin.run(spec, ctx)
    except Exception as exc:
        err_console.print(f"[red]Replay failed:[/red] {exc}")
        err_console.print("[red]" + traceback.format_exc() + "[/red]")
        raise typer.Exit(code=1) from exc

    out_path, _content_hash = _write_envelope(new_envelope, output_dir)
    _print_replay_summary(source, envelope_path, new_envelope, out_path)
