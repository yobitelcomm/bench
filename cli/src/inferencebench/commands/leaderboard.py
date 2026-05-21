"""``bench leaderboard`` — build a static leaderboard from local envelopes.

Wires the ``inferencebench-leaderboard`` package as the rendering engine.
The fetch-from-yobitelcomm.github.io browse mode is Phase 2+; the local
build path is what ships in Phase 1.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

console = Console()
err_console = Console(stderr=True)


def leaderboard(
    category: Annotated[
        str,
        typer.Argument(
            help=(
                "Category id (suite_id) to filter. Omit to render all categories. "
                "Ignored in --build mode."
            ),
        ),
    ] = "",
    envelopes_dir: Annotated[
        Path | None,
        typer.Option(
            "--envelopes",
            "-i",
            help="Directory of signed envelope JSON files. Required with --build.",
        ),
    ] = None,
    out_dir: Annotated[
        Path | None,
        typer.Option(
            "--out",
            "-o",
            help="Destination directory for the rendered static site.",
        ),
    ] = None,
    build: Annotated[
        bool,
        typer.Option(
            "--build/--no-build",
            help="Render a static site from --envelopes into --out.",
        ),
    ] = False,
    base_url: Annotated[
        str,
        typer.Option(
            "--base-url",
            help="Base URL prefix for generated links (e.g. '/bench/' for GH Pages).",
        ),
    ] = "/",
) -> None:
    """Render a leaderboard from local signed envelopes.

    Phase 1 only supports local rendering (``--build``). Hosted-leaderboard
    browse mode (fetching from https://yobitelcomm.github.io/bench) is
    deferred to Phase 2+.
    """
    if not build:
        err_console.print(
            "[yellow]Hosted browse mode is Phase 2+.[/yellow] Pass [bold]--build[/bold] "
            "with [bold]--envelopes <dir>[/bold] [bold]--out <dir>[/bold] to render a "
            "static site from local envelopes."
        )
        raise typer.Exit(code=2)

    if envelopes_dir is None or out_dir is None:
        err_console.print("[red]--build requires both --envelopes <dir> and --out <dir>.[/red]")
        raise typer.Exit(code=2)
    if not envelopes_dir.exists() or not envelopes_dir.is_dir():
        err_console.print(f"[red]Envelopes directory not found:[/red] {envelopes_dir}")
        raise typer.Exit(code=2)

    try:
        from inferencebench_leaderboard import render_site
    except ImportError as exc:
        err_console.print(
            "[red]inferencebench-leaderboard is not installed.[/red] "
            "Install it: [bold]pip install inferencebench-leaderboard[/bold]"
        )
        raise typer.Exit(code=2) from exc

    out_dir.mkdir(parents=True, exist_ok=True)
    result = render_site(envelopes_dir, out_dir, base_url=base_url)

    table = Table(title="Leaderboard render summary", show_header=True)
    table.add_column("metric", style="bold")
    table.add_column("value", style="cyan")
    table.add_row("envelopes loaded", str(result.envelopes_loaded))
    table.add_row("envelopes skipped", str(result.envelopes_skipped))
    table.add_row("categories", ", ".join(sorted(result.categories)) or "(none)")
    table.add_row("output", str(out_dir.resolve()))
    console.print(table)
    if category:
        console.print(
            f"[dim]Category filter [bold]{category}[/bold] ignored — the renderer emits "
            "all categories.[/dim]"
        )
