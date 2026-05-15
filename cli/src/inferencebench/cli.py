"""Typer application root for the ``bench`` CLI.

Subcommands live in ``inferencebench.commands``. Plugin discovery via
``importlib.metadata.entry_points`` group ``inferencebench.plugins``.
"""

from __future__ import annotations

from importlib import metadata
from typing import Annotated

import typer
from rich.console import Console

from inferencebench._logging import configure_logging
from inferencebench.commands import (
    compare,
    cost,
    doctor,
    leaderboard,
    plugin,
    publish,
    run,
    verify,
)

__version__ = "0.0.0"

console = Console()
err_console = Console(stderr=True)

app = typer.Typer(
    name="bench",
    help="InferenceBench Suite — vendor-neutral, signed-envelope AI benchmarks.",
    no_args_is_help=True,
    add_completion=False,
    rich_markup_mode="rich",
)


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"[bold]bench[/bold] {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            help="Show version and exit.",
            callback=_version_callback,
            is_eager=True,
        ),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Verbose logging (DEBUG level)."),
    ] = False,
) -> None:
    """Root CLI callback. Configures logging before dispatching to subcommands."""
    configure_logging("DEBUG" if verbose else "INFO")


# Register subcommands. Each module exports an `app: typer.Typer`.
app.add_typer(run.app, name="run", help="Run a benchmark and produce a signed envelope.")
app.add_typer(compare.app, name="compare", help="Compare benchmark runs (Pareto frontier).")
app.add_typer(publish.app, name="publish", help="Publish a signed envelope (HF Hub, local).")
app.add_typer(verify.app, name="verify", help="Verify a signed envelope's signature + content.")
app.add_typer(leaderboard.app, name="leaderboard", help="Browse public leaderboards.")
app.add_typer(doctor.app, name="doctor", help="Diagnose hardware health before benchmarking.")
app.add_typer(cost.app, name="cost", help="Compare model cost across providers.")
app.add_typer(plugin.app, name="plugin", help="Manage benchmark plugins.")


@app.command("plugins")
def list_plugins() -> None:
    """List installed plugins (shorthand for ``bench plugin list``)."""
    try:
        eps = metadata.entry_points(group="inferencebench.plugins")
    except TypeError:
        # Older Python compat — shouldn't trigger on 3.12 but be safe
        eps = metadata.entry_points().get("inferencebench.plugins", [])  # type: ignore[attr-defined]

    if not eps:
        console.print("[yellow]No plugins installed.[/yellow]")
        console.print("Install one: [bold]pip install inferencebench-llm[/bold]")
        return

    console.print("[bold]Installed plugins:[/bold]")
    for ep in eps:
        console.print(f"  • [cyan]{ep.name}[/cyan]   → {ep.value}")


if __name__ == "__main__":
    app()
