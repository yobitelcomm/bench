"""``bench doctor`` — diagnose hardware health before benchmarking."""

from __future__ import annotations

from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from inferencebench.harness.doctor import CheckStatus, run_diagnostic
from inferencebench.harness.fingerprint import collect_hardware_fingerprint

console = Console()
err_console = Console(stderr=True)


def _render_slo_table() -> None:
    """Print the detected hardware class + resolved ``llm.standard`` SLO row.

    Lives in the CLI (rather than the plugin) so users get the same view
    whether or not a benchmark has been run; uses the plugin's profile
    module directly.
    """
    from inferencebench_llm.plugin import _SLO_TEMPLATES
    from inferencebench_llm.slo_profiles import (
        classify,
        format_resolved,
        scale_slos,
    )

    hw_fp = collect_hardware_fingerprint()
    hw_class = classify(hw_fp)
    base = _SLO_TEMPLATES["llm.standard"]
    resolved = scale_slos(base, hw_class)
    resolved_str = format_resolved(resolved)

    table = Table(title="SLO template (llm.standard)", show_lines=False)
    table.add_column("Field", style="cyan")
    table.add_column("Value")
    table.add_row("Hardware class", hw_class.key)
    table.add_row("Description", hw_class.description)
    table.add_row("ttft multiplier", f"{hw_class.ttft_mult}x")
    table.add_row("tpot multiplier", f"{hw_class.tpot_mult}x")
    table.add_row("total multiplier", f"{hw_class.total_mult}x")
    table.add_row("Resolved thresholds", resolved_str)
    console.print(table)


def doctor(
    strict: Annotated[
        bool,
        typer.Option(
            "--strict",
            help="Refuse if any check returns FAIL or WARN. Default fails only on FAIL.",
        ),
    ] = False,
    show_slo: Annotated[
        bool,
        typer.Option(
            "--show-slo",
            help="Also print the detected hardware class + resolved llm.standard SLO.",
        ),
    ] = False,
) -> None:
    """Run hardware diagnostic. Exit 0 if OK, 1 otherwise."""
    report = run_diagnostic(strict=strict)

    if not report.checks:
        err_console.print("[yellow]No checks ran (no NVIDIA GPUs detected).[/yellow]")
        if show_slo:
            _render_slo_table()
        raise typer.Exit(code=0)

    table = Table(title="Hardware diagnostic", show_lines=False)
    table.add_column("Check", style="cyan")
    table.add_column("Status", style="bold")
    table.add_column("Detail")

    style_for = {
        CheckStatus.PASS: "[green]PASS[/green]",
        CheckStatus.WARN: "[yellow]WARN[/yellow]",
        CheckStatus.FAIL: "[red]FAIL[/red]",
        CheckStatus.SKIP: "[dim]SKIP[/dim]",
    }
    for check in report.checks:
        table.add_row(check.name, style_for[check.status], check.detail)
    console.print(table)

    if report.ok:
        mode = " (strict)" if strict else ""
        console.print(f"[green]OK[/green] — all checks passed{mode}.")
        if show_slo:
            _render_slo_table()
        raise typer.Exit(code=0)

    if show_slo:
        _render_slo_table()
    err_console.print(
        f"[red]REFUSED[/red] — {report.fail_count} FAIL"
        + (f", {report.warn_count} WARN" if strict and report.warn_count else "")
        + ". Resolve hardware issues before benchmarking."
    )
    raise typer.Exit(code=1)
