"""``bench doctor`` — diagnose hardware health before benchmarking."""

from __future__ import annotations

from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from inferencebench.harness.doctor import CheckStatus, run_diagnostic

console = Console()
err_console = Console(stderr=True)


def doctor(
    strict: Annotated[
        bool,
        typer.Option(
            "--strict",
            help="Refuse if any check returns FAIL or WARN. Default fails only on FAIL.",
        ),
    ] = False,
) -> None:
    """Run hardware diagnostic. Exit 0 if OK, 1 otherwise."""
    report = run_diagnostic(strict=strict)

    if not report.checks:
        err_console.print("[yellow]No checks ran (no NVIDIA GPUs detected).[/yellow]")
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
        raise typer.Exit(code=0)

    err_console.print(
        f"[red]REFUSED[/red] — {report.fail_count} FAIL"
        + (f", {report.warn_count} WARN" if strict and report.warn_count else "")
        + ". Resolve hardware issues before benchmarking."
    )
    raise typer.Exit(code=1)
