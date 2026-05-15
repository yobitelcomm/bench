"""``bench plugin`` — manage benchmark plugins.

Subcommands: list, init, install, info.
Plugin discovery via Python entry points.
"""

from __future__ import annotations

from importlib import metadata
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(no_args_is_help=True)
console = Console()
err_console = Console(stderr=True)


def _discover_plugins() -> list[metadata.EntryPoint]:
    """Return all registered plugin entry points (Phase 1: typically empty)."""
    try:
        return list(metadata.entry_points(group="inferencebench.plugins"))
    except TypeError:
        # Pre-3.10 compat path — shouldn't trigger on 3.12
        return list(metadata.entry_points().get("inferencebench.plugins", []))  # type: ignore[attr-defined]


@app.command("list")
def list_plugins() -> None:
    """List installed plugins."""
    eps = _discover_plugins()
    if not eps:
        console.print("[yellow]No plugins installed.[/yellow]")
        console.print("Install one: [bold]pip install inferencebench-llm[/bold]")
        return

    table = Table(title="Installed plugins")
    table.add_column("Name", style="cyan")
    table.add_column("Module", style="dim")
    table.add_column("Distribution", style="green")
    for ep in eps:
        dist = ep.dist.name if ep.dist else "?"
        table.add_row(ep.name, ep.value, dist)
    console.print(table)


@app.command("init")
def init_plugin(
    name: Annotated[str, typer.Argument(help="New plugin name (e.g. 'voice').")],
    kind: Annotated[str, typer.Option("--kind", help="perf, quality, or both.")] = "both",
    modality: Annotated[str, typer.Option("--modality", help="llm, voice, video, 3d, ...")] = "",
) -> None:
    """Scaffold a new plugin package.

    Phase 1 stub — ticket 0028 will wire the actual scaffolder.
    """
    err_console.print(
        f"[yellow][stub][/yellow] bench plugin init [bold]{name}[/bold] "
        f"--kind {kind} --modality {modality or '<none>'} — "
        "not yet implemented in v0.0.0 (ticket 0028)."
    )


@app.command("install")
def install_plugin(
    package: Annotated[str, typer.Argument(help="Plugin package name (e.g. inferencebench-llm).")],
) -> None:
    """Install a plugin from PyPI.

    Phase 1 stub — ticket 0028. (User can already do this with pip directly.)
    """
    err_console.print(
        f"[yellow][stub][/yellow] bench plugin install [bold]{package}[/bold] — "
        "not yet implemented in v0.0.0 (ticket 0028). Use [bold]pip install[/bold] instead."
    )


@app.command("info")
def info_plugin(
    name: Annotated[str, typer.Argument(help="Plugin name to introspect.")],
) -> None:
    """Show information about an installed plugin."""
    eps = _discover_plugins()
    matched = [ep for ep in eps if ep.name == name]
    if not matched:
        err_console.print(f"[red]Plugin not found:[/red] {name}")
        err_console.print(f"Installed: {[ep.name for ep in eps] or 'none'}")
        raise typer.Exit(code=1)

    ep = matched[0]
    console.print(f"[bold]{ep.name}[/bold]")
    console.print(f"  module:        {ep.value}")
    if ep.dist:
        console.print(f"  distribution:  {ep.dist.name} {ep.dist.version}")
        meta = ep.dist.metadata
        if meta:
            console.print(f"  summary:       {meta.get('Summary', '<none>')}")
            console.print(f"  homepage:      {meta.get('Home-page', '<none>')}")
