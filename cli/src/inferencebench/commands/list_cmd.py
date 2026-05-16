"""``bench list`` — top-level catalogue of every benchmark across every plugin.

Where ``bench plugins`` (and ``bench plugin list``) enumerate installed plugin
packages, this command goes one level deeper: it walks each plugin, calls
``plugin.list_benchmarks()``, and renders a single table so users can scan
the full menu without invoking each plugin individually.
"""

from __future__ import annotations

from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from inferencebench.commands.run import _entry_points

console = Console()
err_console = Console(stderr=True)


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _spec_to_dict(spec: Any) -> dict[str, Any]:  # noqa: ANN401
    """Best-effort serialise a plugin BenchmarkSpec to a JSON-safe dict.

    Different plugins may use different spec models; we lean on Pydantic
    when it's available and otherwise fall back to attribute introspection.
    """
    dump = getattr(spec, "model_dump", None)
    if callable(dump):
        try:
            return dict(dump(mode="json"))
        except TypeError:
            return dict(dump())
    return {
        k: getattr(spec, k, None)
        for k in (
            "benchmark_id",
            "suite_version",
            "description",
            "modality",
            "kind",
            "dataset",
            "driver",
            "slo_template",
            "metrics",
        )
    }


def _driver_label(spec: Any) -> str:  # noqa: ANN401
    """Extract a short driver label (``open_loop`` / ``closed_loop``) from a spec."""
    driver = getattr(spec, "driver", None)
    if driver is None:
        return "-"
    return str(getattr(driver, "type", driver) or "-")


def _dataset_label(spec: Any) -> str:  # noqa: ANN401
    """Extract the dataset id from a spec."""
    dataset = getattr(spec, "dataset", None)
    if dataset is None:
        return "-"
    return str(getattr(dataset, "id", dataset) or "-")


def _description_short(spec: Any, limit: int = 60) -> str:  # noqa: ANN401
    """Return the first ``limit`` characters of the spec description (no newlines)."""
    desc = getattr(spec, "description", "") or ""
    first_line = desc.strip().splitlines()[0] if desc.strip() else ""
    if len(first_line) <= limit:
        return first_line
    return first_line[: limit - 1].rstrip() + "…"


# --------------------------------------------------------------------------- #
# Command                                                                     #
# --------------------------------------------------------------------------- #
def list_benchmarks(
    plugin_filter: Annotated[
        str,
        typer.Option(
            "--plugin",
            help="Filter to a single plugin (e.g. 'llm.inference').",
        ),
    ] = "",
    json_out: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Emit machine-readable JSON instead of a Rich table.",
        ),
    ] = False,
) -> None:
    """List every benchmark across every installed plugin."""
    eps = _entry_points()

    if plugin_filter:
        eps = [ep for ep in eps if ep.name == plugin_filter]
        if not eps:
            err_console.print(
                f"[red]No plugin registered for:[/red] [bold]{plugin_filter}[/bold]"
            )
            raise typer.Exit(code=1)

    if not eps:
        if json_out:
            console.print_json(data={"plugins": {}})
        else:
            console.print("[yellow]No plugins installed.[/yellow]")
            console.print("Install one: [bold]pip install inferencebench-llm[/bold]")
        return

    payload: dict[str, Any] = {"plugins": {}}
    table_rows: list[tuple[str, Any]] = []

    for ep in eps:
        try:
            plugin_cls = ep.load()
            plugin = plugin_cls()
            specs = list(plugin.list_benchmarks())
        except Exception as exc:  # pragma: no cover - defensive
            err_console.print(
                f"[yellow]warning:[/yellow] failed to introspect plugin "
                f"'{ep.name}': {exc}"
            )
            payload["plugins"][ep.name] = {
                "version": "",
                "error": str(exc),
                "benchmarks": [],
            }
            continue

        payload["plugins"][ep.name] = {
            "version": getattr(plugin, "version", "") or "",
            "benchmarks": [_spec_to_dict(s) for s in specs],
        }
        for spec in specs:
            table_rows.append((ep.name, spec))

    if json_out:
        console.print_json(data=payload)
        return

    table = Table(title="Available benchmarks")
    table.add_column("Plugin", style="cyan", no_wrap=True)
    table.add_column("Benchmark ID", style="bold")
    table.add_column("Modality")
    table.add_column("Kind")
    table.add_column("Driver")
    table.add_column("Dataset")
    table.add_column("Description")

    for plugin_name, spec in table_rows:
        table.add_row(
            plugin_name,
            str(getattr(spec, "benchmark_id", "-")),
            str(getattr(spec, "modality", "-") or "-"),
            str(getattr(spec, "kind", "-") or "-"),
            _driver_label(spec),
            _dataset_label(spec),
            _description_short(spec),
        )

    if not table_rows:
        console.print("[yellow]No benchmarks exposed by installed plugins.[/yellow]")
        return

    console.print(table)
