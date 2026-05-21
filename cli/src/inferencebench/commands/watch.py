"""``bench watch`` — continuously rebuild a leaderboard as envelopes arrive.

Polls an envelopes directory at a configurable interval and re-renders the
static leaderboard whenever the set of envelope files (or any file's mtime)
changes. Pairs naturally with a long-running ``bench run --sweep`` so the
hosted leaderboard stays fresh without manual ``bench leaderboard --build``
invocations.

The polling implementation is deliberately pure-Python: portable across
Linux/macOS/Windows, dependency-free, and good enough at the default 5s
cadence. inotify/fsevents would be a marginal win for the cost of an extra
dependency.
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

console = Console()
err_console = Console(stderr=True)


def _snapshot(envelopes_dir: Path) -> dict[str, float]:
    """Return ``{abs_path: mtime}`` for every ``*.json`` under ``envelopes_dir``."""
    snapshot: dict[str, float] = {}
    for path in envelopes_dir.rglob("*.json"):
        if not path.is_file():
            continue
        try:
            snapshot[str(path.resolve())] = path.stat().st_mtime
        except OSError:
            # File vanished between rglob and stat — skip it; next poll picks it up.
            continue
    return snapshot


def watch(
    envelopes_dir: Annotated[
        Path,
        typer.Argument(
            help="Directory of signed envelope JSON files to watch (recursive).",
        ),
    ],
    out_dir: Annotated[
        Path,
        typer.Option(
            "--out",
            "-o",
            help="Destination directory for the rendered static site.",
        ),
    ],
    interval_s: Annotated[
        float,
        typer.Option(
            "--interval-s",
            help="Polling interval in seconds.",
        ),
    ] = 5.0,
    base_url: Annotated[
        str,
        typer.Option(
            "--base-url",
            help="Base URL prefix for generated links (e.g. '/bench/' for GH Pages).",
        ),
    ] = "/",
    max_iterations: Annotated[
        int,
        typer.Option(
            "--max-iterations",
            help="Stop after N polls (0 = unlimited). Primarily for tests/CI.",
        ),
    ] = 0,
    quiet: Annotated[
        bool,
        typer.Option(
            "--quiet",
            "-q",
            help="Suppress 'no changes' lines between rebuilds.",
        ),
    ] = False,
) -> None:
    """Watch ``envelopes_dir`` and rebuild the leaderboard on changes.

    Renders an initial site on the first iteration (even if the directory is
    empty), then polls every ``--interval-s`` seconds for changes — added,
    removed, or modified ``*.json`` files. On change, calls ``render_site``
    and prints a one-line Rich summary. Exits 0 on Ctrl-C or when
    ``--max-iterations`` is reached; exits 1 if ``render_site`` ever raised.
    """
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

    previous: dict[str, float] = {}
    iteration = 0
    had_failure = False

    try:
        while True:
            iteration += 1
            current = _snapshot(envelopes_dir)
            should_render = iteration == 1 or current != previous

            if should_render:
                try:
                    result = render_site(envelopes_dir, out_dir, base_url=base_url)
                except Exception as exc:  # keep watching after any render failure
                    had_failure = True
                    err_console.print(
                        f"[yellow]render_site failed:[/yellow] {exc!s} "
                        f"at {datetime.now().strftime('%H:%M:%S')}"
                    )
                else:
                    console.print(
                        f"[green]rebuilt site:[/green] "
                        f"{result.envelopes_loaded} envelopes, "
                        f"{result.envelopes_skipped} skipped, "
                        f"{len(result.categories)} categories "
                        f"at {datetime.now().strftime('%H:%M:%S')}"
                    )
                previous = current
            elif not quiet:
                console.print(f"[dim]no changes at {datetime.now().strftime('%H:%M:%S')}[/dim]")

            if max_iterations and iteration >= max_iterations:
                break

            time.sleep(interval_s)
    except KeyboardInterrupt:
        console.print("[dim]stopping[/dim]")

    if had_failure:
        raise typer.Exit(code=1)
