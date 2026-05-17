"""``bench spec`` — schema-check user-supplied benchmark YAML files.

Custom plugins (scaffolded by ``bench plugin init``) and custom benchmark
specs need a quick way to validate a YAML file against the BenchmarkSpec
schema each installed plugin exposes, *before* attempting to run anything.
The fall-back path of "run bench and squint at pydantic errors" is slow;
this command short-circuits it.

Three subcommands:

* ``bench spec validate <yaml-file>`` — try each installed plugin's
  ``BenchmarkSpec`` import. A spec is considered valid iff at least one
  plugin accepts it. Prints a per-plugin Rich table. Exit 0 if any plugin
  accepts; exit 1 if none do.
* ``bench spec show <yaml-file>`` — validate, then pretty-print the parsed
  spec as a Rich tree so users can confirm their YAML parsed the way they
  expected. Exit 0 on success, 1 on validation failure.
* ``bench spec lint <yaml-file>`` — validate + soft heuristics (short
  duration, suspiciously high RPS, missing dataset id / description).
  Lints are warnings, not errors; exit 0 always.

The plugin discovery + schema-resolution helpers come from
:mod:`inferencebench.commands.run` so we always agree with ``bench run``
on which plugins exist and what their top-level types look like.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import typer
import yaml
from pydantic import BaseModel, ValidationError
from rich.console import Console
from rich.table import Table
from rich.tree import Tree

from inferencebench.commands.run import _entry_points

if TYPE_CHECKING:
    from importlib.metadata import EntryPoint

app = typer.Typer(no_args_is_help=True)
console = Console()
err_console = Console(stderr=True)


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file. Exit 2 if missing or invalid."""
    if not path.is_file():
        err_console.print(f"[red]Spec file not found:[/red] {path}")
        raise typer.Exit(code=2)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        err_console.print(f"[red]Failed to parse YAML:[/red] {exc}")
        raise typer.Exit(code=2) from exc
    if not isinstance(raw, dict):
        err_console.print(
            f"[red]Spec file must be a mapping at the top level, got "
            f"{type(raw).__name__}.[/red]"
        )
        raise typer.Exit(code=2)
    return raw


def _resolve_benchmark_spec(ep: EntryPoint) -> type[BaseModel]:
    """Return the ``BenchmarkSpec`` class re-exported by a plugin's top-level.

    Mirrors the ``_resolve_plugin_schemas`` convention in
    :mod:`inferencebench.commands.run` — plugins are expected to expose
    ``BenchmarkSpec`` at their top-level package.
    """
    module_path = ep.value.split(":")[0]
    top_pkg = module_path.split(".")[0]
    pkg = importlib.import_module(top_pkg)
    try:
        spec_cls = pkg.BenchmarkSpec
    except AttributeError as exc:
        msg = (
            f"Plugin '{ep.name}' (package '{top_pkg}') does not expose "
            "BenchmarkSpec at its top level."
        )
        raise RuntimeError(msg) from exc
    if not isinstance(spec_cls, type) or not issubclass(spec_cls, BaseModel):
        msg = (
            f"Plugin '{ep.name}' exposes BenchmarkSpec but it is not a "
            "pydantic BaseModel subclass."
        )
        raise RuntimeError(msg)
    return spec_cls


def _first_error(exc: ValidationError) -> str:
    """Return a short one-line summary of the first error in a ValidationError."""
    errors = exc.errors()
    if not errors:
        return str(exc)
    err = errors[0]
    loc = ".".join(str(part) for part in err.get("loc", ()))
    msg = err.get("msg", "validation error")
    if loc:
        return f"{loc}: {msg}"
    return msg


def _validate_against_all(
    raw: dict[str, Any],
) -> tuple[list[tuple[str, bool, str, BaseModel | None]], BaseModel | None]:
    """Try each installed plugin's BenchmarkSpec against ``raw``.

    Returns:
        (rows, accepted) where ``rows`` is a list of
        ``(plugin_name, ok, reason, parsed-or-None)`` tuples and ``accepted``
        is the first parsed BenchmarkSpec instance, or ``None`` if every
        plugin rejected the YAML.
    """
    eps = _entry_points()
    rows: list[tuple[str, bool, str, BaseModel | None]] = []
    accepted: BaseModel | None = None
    for ep in eps:
        try:
            spec_cls = _resolve_benchmark_spec(ep)
        except RuntimeError as exc:
            rows.append((ep.name, False, f"schema-resolution: {exc}", None))
            continue
        try:
            parsed = spec_cls.model_validate(raw)
        except ValidationError as exc:
            rows.append((ep.name, False, _first_error(exc), None))
            continue
        rows.append((ep.name, True, "", parsed))
        if accepted is None:
            accepted = parsed
    return rows, accepted


# --------------------------------------------------------------------------- #
# Subcommand: validate                                                        #
# --------------------------------------------------------------------------- #
@app.command("validate")
def validate(
    yaml_file: Annotated[
        Path,
        typer.Argument(help="Path to a benchmark spec YAML file to validate."),
    ],
) -> None:
    """Validate a YAML benchmark spec against every installed plugin's schema.

    A spec is accepted if *any* installed plugin's ``BenchmarkSpec`` model
    parses it without error. The per-plugin breakdown is printed as a Rich
    table; the exit code is 0 if at least one plugin accepted, 1 otherwise.
    """
    raw = _load_yaml(yaml_file)
    rows, accepted = _validate_against_all(raw)

    table = Table(title=f"Spec validation: {yaml_file.name}")
    table.add_column("plugin", style="cyan", no_wrap=True)
    table.add_column("result", justify="center")
    table.add_column("reason", style="red")
    if not rows:
        table.add_row("[yellow]<none>[/yellow]", "-", "no plugins installed")
    for plugin_name, ok, reason, _parsed in rows:
        marker = "[bold green]✓ valid[/]" if ok else "[bold red]✗ rejected[/]"
        table.add_row(plugin_name, marker, "" if ok else reason[:80])
    console.print(table)

    if accepted is None:
        err_console.print(
            "[red]No installed plugin accepted this spec.[/red] "
            "Check the field names + types against `bench schema --target benchmark-spec`."
        )
        raise typer.Exit(code=1)

    console.print(
        "[green]ok[/green] spec validates under at least one plugin schema."
    )


# --------------------------------------------------------------------------- #
# Subcommand: show                                                            #
# --------------------------------------------------------------------------- #
@app.command("show")
def show(
    yaml_file: Annotated[
        Path,
        typer.Argument(help="Path to a benchmark spec YAML file to pretty-print."),
    ],
) -> None:
    """Validate then pretty-print a benchmark spec as a Rich tree.

    Useful for confirming a YAML file parsed exactly the way the author
    expected — Pydantic defaults, type coercions, and missing fields all
    show up clearly in the tree.
    """
    raw = _load_yaml(yaml_file)
    _rows, accepted = _validate_against_all(raw)
    if accepted is None:
        err_console.print(
            "[red]Spec failed validation under every installed plugin.[/red] "
            "Run [bold]bench spec validate[/bold] for per-plugin reasons."
        )
        raise typer.Exit(code=1)

    tree = Tree(f"[bold]{yaml_file.name}[/bold] (parsed BenchmarkSpec)")
    _build_tree(tree, accepted.model_dump(mode="json"))
    console.print(tree)


def _build_tree(node: Tree, value: Any) -> None:  # noqa: ANN401 — recursive over JSON-shaped data
    """Recursively render a JSON-shaped dict/list/scalar into a Rich tree node."""
    if isinstance(value, dict):
        for key in sorted(value.keys()):
            sub = value[key]
            if isinstance(sub, dict | list):
                branch = node.add(f"[cyan]{key}[/cyan]")
                _build_tree(branch, sub)
            else:
                node.add(f"[cyan]{key}[/cyan] = [bold]{_fmt_scalar(sub)}[/bold]")
    elif isinstance(value, list):
        if not value:
            node.add("[dim](empty list)[/dim]")
            return
        for idx, item in enumerate(value):
            if isinstance(item, dict | list):
                branch = node.add(f"[dim][{idx}][/dim]")
                _build_tree(branch, item)
            else:
                node.add(f"[dim][{idx}][/dim] = [bold]{_fmt_scalar(item)}[/bold]")
    else:
        node.add(f"[bold]{_fmt_scalar(value)}[/bold]")


def _fmt_scalar(value: Any) -> str:  # noqa: ANN401 — JSON scalars only
    """Render a scalar leaf as a short string, preserving ``None``."""
    if value is None:
        return "[dim]None[/dim]"
    if isinstance(value, str):
        return repr(value) if not value else value
    return str(value)


# --------------------------------------------------------------------------- #
# Subcommand: lint                                                            #
# --------------------------------------------------------------------------- #
@app.command("lint")
def lint(
    yaml_file: Annotated[
        Path,
        typer.Argument(help="Path to a benchmark spec YAML file to lint."),
    ],
) -> None:
    """Validate + soft heuristic checks for a benchmark spec.

    Lints are warnings, not errors — this command always exits 0. The
    heuristics are intentionally loose; the goal is "things authors would
    probably want to know" rather than a strict policy.
    """
    raw = _load_yaml(yaml_file)
    _rows, accepted = _validate_against_all(raw)

    warnings = _collect_lint_warnings(accepted, raw)

    table = Table(title=f"Spec lint: {yaml_file.name}")
    table.add_column("check", style="cyan", no_wrap=True)
    table.add_column("warning", style="yellow")
    if accepted is None:
        table.add_row(
            "schema",
            "spec did not validate under any installed plugin; "
            "lints below are best-effort from the raw YAML.",
        )
    if not warnings:
        table.add_row("ok", "no warnings")
    for check, message in warnings:
        table.add_row(check, message)
    console.print(table)


def _collect_lint_warnings(
    spec: BaseModel | None, raw: dict[str, Any]
) -> list[tuple[str, str]]:
    """Return ``(check_name, warning)`` pairs for every soft heuristic that fires.

    Reads from the parsed spec when available (so we benefit from pydantic's
    type coercion + defaults), otherwise falls back to the raw YAML dict.
    """
    warnings: list[tuple[str, str]] = []
    data: dict[str, Any] = spec.model_dump(mode="json") if spec is not None else dict(raw)

    duration = _extract_duration_s(data)
    if duration is not None and duration < 30:
        warnings.append(
            (
                "duration_s",
                f"short duration ({duration}s) may not reach steady state",
            )
        )

    rps_value = _extract_first_rps(data)
    if rps_value is not None and rps_value > 1000:
        warnings.append(
            (
                "rps",
                f"very high RPS ({rps_value:g}), ensure your engine can sustain it",
            )
        )

    dataset_id = _extract_dataset_id(data)
    if not dataset_id:
        warnings.append(
            ("dataset.id", "dataset.id is empty; results will be hard to interpret"),
        )

    description = data.get("description")
    if not isinstance(description, str) or not description.strip():
        warnings.append(
            (
                "description",
                "description is empty; consider adding one for discoverability",
            )
        )

    return warnings


def _extract_duration_s(data: dict[str, Any]) -> int | None:
    """Pull ``driver.duration_s`` out of the spec dict, ``None`` if missing/non-int."""
    driver = data.get("driver")
    if not isinstance(driver, dict):
        return None
    value = driver.get("duration_s")
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _extract_first_rps(data: dict[str, Any]) -> float | None:
    """Pull ``driver.rps[0]`` out of the spec dict, ``None`` if missing/non-numeric."""
    driver = data.get("driver")
    if not isinstance(driver, dict):
        return None
    rps = driver.get("rps")
    if not isinstance(rps, list) or not rps:
        return None
    first = rps[0]
    if isinstance(first, bool) or not isinstance(first, int | float):
        return None
    return float(first)


def _extract_dataset_id(data: dict[str, Any]) -> str:
    """Pull ``dataset.id`` out of the spec dict, ``""`` if missing/non-string."""
    dataset = data.get("dataset")
    if not isinstance(dataset, dict):
        return ""
    value = dataset.get("id")
    if not isinstance(value, str):
        return ""
    return value.strip()
