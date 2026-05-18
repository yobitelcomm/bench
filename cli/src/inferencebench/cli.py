"""Typer application root for the ``bench`` CLI.

Single-action subcommands live in ``inferencebench.commands.*`` as plain
functions and get registered on the root ``app``. The ``plugin`` command,
which has multiple sub-subcommands, stays as a sub-Typer.
"""

from __future__ import annotations

from importlib import metadata
from typing import Annotated

import typer
from rich.console import Console

from inferencebench._logging import configure_logging
from inferencebench.commands import (
    attest,
    audit,
    bundle,
    cache,
    ci,
    cluster,
    compare,
    cost,
    coverage,
    dashboard,
    diff,
    doctor,
    export,
    fetch,
    fixtures,
    history,
    leaderboard,
    list_cmd,
    matrix,
    plugin,
    profile,
    publish,
    replay,
    run,
    schema_cmd,
    server,
    spec,
    summary,
    tour,
    verify,
    watch,
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
        raise typer.Exit


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


# Single-action subcommands — registered directly so option/positional parsing
# behaves correctly (sub-Typer with invoke_without_command=True has a parsing
# quirk that breaks positional-then-option args).
app.command(name="run", help="Run a benchmark and produce a signed envelope.")(run.run)
app.command(
    name="audit",
    help="Verify every envelope in a directory and report failures.",
)(audit.audit)
app.command(
    name="attest",
    help="Render a third-party-readable attestation slip from an envelope.",
)(attest.attest)
# `bundle` has subcommands (create/extract) → sub-Typer
app.add_typer(
    bundle.app,
    name="bundle",
    help="Pack/unpack a one-file shareable envelope bundle (.bundle.zip).",
)
# `cache` has subcommands (list/clear/path) → sub-Typer
app.add_typer(cache.app, name="cache", help="Manage the local envelope fetch cache.")
# `ci` has subcommands (init/validate) → sub-Typer
app.add_typer(
    ci.app,
    name="ci",
    help="Generate or validate a GitHub Actions regression-check workflow.",
)
# `cluster` has subcommands (run/status/sync) → sub-Typer
app.add_typer(
    cluster.app,
    name="cluster",
    help="Runner-side coordinator: dispatch matrix runs + ship envelopes to a bench server.",
)
app.command(name="compare", help="Compare benchmark runs (Pareto frontier).")(compare.compare)
app.command(name="fetch", help="Fetch a signed envelope from a remote URI.")(fetch.fetch)
# `fixtures` has subcommands (list/fetch/path/clear) → sub-Typer
app.add_typer(
    fixtures.app,
    name="fixtures",
    help="Fetch real public dataset fixtures (FLORES, HumanEval, GSM8K, ...).",
)
app.command(
    name="history",
    help="Time-series view of one metric across runs.",
)(history.history)
app.command(
    name="profile",
    help="Re-run a benchmark with high-frequency telemetry for diagnosis.",
)(profile.profile)
app.command(name="publish", help="Publish a signed envelope (HF Hub, local).")(publish.publish)
app.command(name="replay", help="Replay a benchmark from an existing envelope.")(replay.replay)
app.command(
    name="schema",
    help="Emit JSON Schema for envelopes / benchmark specs / mirror index.",
)(schema_cmd.schema)
app.command(name="verify", help="Verify a signed envelope's signature + content.")(verify.verify)
app.command(name="leaderboard", help="Browse public leaderboards.")(leaderboard.leaderboard)
app.command(
    name="list",
    help="List every benchmark across every installed plugin.",
)(list_cmd.list_benchmarks)
app.command(
    name="matrix",
    help="Run one benchmark across multiple endpoints (multi-vendor matrix).",
)(matrix.matrix)
app.command(name="doctor", help="Diagnose hardware health before benchmarking.")(doctor.doctor)
app.command(
    name="export",
    help="Export an envelope as markdown / CSV / Slack snippet.",
)(export.export)
app.command(name="cost", help="Compare model cost across providers.")(cost.cost)
app.command(
    name="coverage",
    help="Report metric completeness per envelope (plugin-aware EXPECTED_METRICS).",
)(coverage.coverage)
app.command(
    name="dashboard",
    help="Serve a live leaderboard over HTTP with auto-rescan on each request.",
)(dashboard.dashboard)
app.command(name="diff", help="Per-metric delta between two envelopes.")(diff.diff)
app.command(
    name="server",
    help="Run a minimal HTTP API for distributed envelope ingestion.",
)(server.server)
# `spec` has subcommands (validate/show/lint) → sub-Typer
app.add_typer(
    spec.app,
    name="spec",
    help="Validate / show / lint user-supplied benchmark spec YAML files.",
)
app.command(name="summary", help="Summarise envelopes in a directory or file.")(summary.summary)
app.command(
    name="tour",
    help="Interactive end-to-end install-validation walkthrough.",
)(tour.tour)
app.command(
    name="watch",
    help="Watch an envelopes directory and rebuild the leaderboard on changes.",
)(watch.watch)

# `plugin` has subcommands (list/init/install/info) → sub-Typer
app.add_typer(plugin.app, name="plugin", help="Manage benchmark plugins.")


@app.command("plugins", help="List installed plugins (shorthand for ``bench plugin list``).")
def list_plugins() -> None:
    """List installed plugins (shorthand)."""
    try:
        eps = metadata.entry_points(group="inferencebench.plugins")
    except TypeError:
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
