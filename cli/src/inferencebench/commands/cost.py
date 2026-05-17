"""``bench cost`` — compare model cost across providers.

Phase 1 implementation (ticket 0027). Reads the in-process pricing registry
shipped with ``inferencebench-llm`` and renders a per-provider price table for
one model, with a configurable input/output blend.
"""

from __future__ import annotations

import difflib
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer
from rich.console import Console
from rich.table import Table

if TYPE_CHECKING:
    from inferencebench_llm.pricing import ModelPricing

console = Console()
err_console = Console(stderr=True)


def cost(
    model: Annotated[
        str,
        typer.Argument(
            help="Model id (e.g. llama-4-maverick). Omit when using --validate-prices.",
        ),
    ] = "",
    suite: Annotated[
        str,
        typer.Option(
            "--suite",
            help=(
                "Suite for the cost comparison. Currently informational only — "
                "Phase 1 does not yet wire this to suite-aware pricing."
            ),
        ),
    ] = "intelligence-index",
    providers: Annotated[
        str,
        typer.Option(
            "--providers",
            help="Comma-separated provider list (together,fireworks,groq,...).",
        ),
    ] = "",
    input_token_share: Annotated[
        float,
        typer.Option(
            "--input-token-share",
            help=(
                "Share of input tokens in the blended cost column "
                "(default 0.75 → blended = 0.75*input + 0.25*output)."
            ),
            min=0.0,
            max=1.0,
        ),
    ] = 0.75,
    prices_file: Annotated[
        str,
        typer.Option(
            "--prices-file",
            help=(
                "Path to a custom prices YAML to use instead of the bundled "
                "registry. See plugins/llm-inference/src/inferencebench_llm/"
                "prices.yaml for the expected schema."
            ),
        ),
    ] = "",
    validate_prices: Annotated[
        str,
        typer.Option(
            "--validate-prices",
            help=(
                "Parse the given YAML pricing file, print a validity summary, "
                "and exit. Useful when editing your own prices file. No table "
                "is rendered in this mode."
            ),
        ),
    ] = "",
) -> None:
    """Compare model cost across providers using the bundled pricing registry."""
    if validate_prices:
        _run_validate_prices(validate_prices)
        return

    if not model:
        err_console.print(
            "[red]Missing argument:[/red] MODEL is required unless --validate-prices is used."
        )
        raise typer.Exit(code=2)

    if suite:
        # Informational only; reserved for Phase 2 suite-aware pricing.
        pass

    entries = _resolve_entries(model, providers, prices_file)
    _render_cost_table(entries, model=model, suite=suite, input_token_share=input_token_share)


def _resolve_entries(
    model: str,
    providers: str,
    prices_file: str,
) -> list[ModelPricing]:
    """Collect the matching pricing entries, exiting with a clear error on miss."""
    try:
        from inferencebench_llm.pricing import (
            all_providers,
            load_pricing,
            lookup,
            models_for,
        )
    except ImportError as exc:
        err_console.print(
            "[red]bench cost requires the inferencebench-llm plugin "
            "to be installed.[/red]"
        )
        err_console.print("  pip install inferencebench-llm")
        raise typer.Exit(code=2) from exc

    custom_registry = (
        _load_custom_registry(prices_file, load_pricing) if prices_file else None
    )
    requested = _parse_providers(providers)

    if custom_registry is not None:
        candidate_providers = (
            requested if requested else sorted({p for p, _ in custom_registry})
        )
        entries = _lookup_in(custom_registry, candidate_providers, model)
    else:
        candidate_providers = requested if requested else all_providers()
        entries = [
            priced
            for provider in candidate_providers
            if (priced := lookup(provider, model)) is not None
        ]

    if not entries:
        if custom_registry is not None:
            _emit_not_found_custom(model, custom_registry)
        else:
            _emit_not_found_bundled(model, all_providers, models_for)
        raise typer.Exit(code=1)
    return entries


def _render_cost_table(
    entries: list[ModelPricing],
    *,
    model: str,
    suite: str,
    input_token_share: float,
) -> None:
    """Render the per-provider cost table to stdout."""
    output_share = 1.0 - input_token_share

    rows: list[tuple[ModelPricing, float]] = []
    for entry in entries:
        blended = (
            input_token_share * entry.input_per_million_usd
            + output_share * entry.output_per_million_usd
        )
        rows.append((entry, blended))
    rows.sort(key=lambda r: r[1])

    table = Table(
        title=(
            f"Cost for [bold]{model}[/bold]  "
            f"(blend = {input_token_share:.2f} input + {output_share:.2f} output, "
            f"suite={suite})"
        ),
        show_header=True,
        header_style="bold",
    )
    table.add_column("Provider")
    table.add_column("Input $/Mtok", justify="right")
    table.add_column("Output $/Mtok", justify="right")
    table.add_column(
        f"Blended ({_ratio_label(input_token_share)}) $/Mtok", justify="right"
    )
    table.add_column("Notes")

    for entry, blended in rows:
        table.add_row(
            entry.provider,
            f"${entry.input_per_million_usd:,.2f}",
            f"${entry.output_per_million_usd:,.2f}",
            f"${blended:,.2f}",
            entry.notes or "-",
        )

    console.print(table)


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _parse_providers(raw: str) -> list[str]:
    """Parse the ``--providers`` flag value into a normalized list."""
    return [p.strip().lower() for p in raw.split(",") if p.strip()]


def _ratio_label(input_share: float) -> str:
    """Render the blend ratio as ``a:b`` integer-ish (e.g. 0.75 → ``3:1``)."""
    if input_share <= 0.0:
        return "0:1"
    if input_share >= 1.0:
        return "1:0"
    output_share = 1.0 - input_share
    # Approximate as integer ratio (e.g. 0.75 → 3:1, 0.5 → 1:1) when clean.
    if abs(input_share - 0.75) < 1e-6:
        return "3:1"
    if abs(input_share - 0.5) < 1e-6:
        return "1:1"
    if abs(input_share - 0.25) < 1e-6:
        return "1:3"
    return f"{input_share:.2f}:{output_share:.2f}"


def _load_custom_registry(
    prices_file: str,
    load_pricing: object,  # callable
) -> dict[tuple[str, str], ModelPricing]:
    """Load a user-supplied pricing YAML or exit with a clear error message."""
    path = Path(prices_file)
    if not path.is_file():
        err_console.print(
            f"[red]--prices-file not found:[/red] {path}"
        )
        raise typer.Exit(code=2)
    try:
        registry = load_pricing(path)  # type: ignore[operator]
    except (ValueError, OSError) as exc:
        err_console.print(
            f"[red]Failed to load --prices-file {path}:[/red] {exc}"
        )
        raise typer.Exit(code=2) from exc
    err_console.print(
        f"[yellow]Using custom pricing from {path}[/yellow]"
    )
    # mypy: load_pricing is opaque here, but at runtime returns the right type.
    return registry  # type: ignore[no-any-return]


def _lookup_in(
    registry: dict[tuple[str, str], ModelPricing],
    candidate_providers: list[str],
    model: str,
) -> list[ModelPricing]:
    """Resolve ``model`` against a custom registry, mirroring ``pricing.lookup``."""
    out: list[ModelPricing] = []
    target = model.strip()
    head_tail: tuple[str, str] | None = None
    if "/" in target:
        head, tail = target.split("/", 1)
        head_tail = (head.lower(), tail)
    for provider in candidate_providers:
        key = (provider.lower().strip(), target)
        if key in registry:
            out.append(registry[key])
            continue
        if head_tail is not None:
            entry = registry.get(head_tail)
            if entry is not None and entry.provider == provider:
                out.append(entry)
    return out


def _emit_not_found_bundled(
    model: str,
    all_providers: object,  # callable returning list[str]
    models_for: object,  # callable provider -> list[str]
) -> None:
    """Render a red error + similar-model suggestions for an unknown model."""
    err_console.print(
        f"[red]No pricing entry found for model:[/red] [bold]{model}[/bold]"
    )

    providers_list: list[str] = all_providers()  # type: ignore[operator]
    all_models = sorted(
        {m for provider in providers_list for m in models_for(provider)}  # type: ignore[operator]
    )
    _print_suggestions(model, all_models)


def _emit_not_found_custom(
    model: str,
    registry: dict[tuple[str, str], ModelPricing],
) -> None:
    """Same as :func:`_emit_not_found_bundled` but for a user-supplied registry."""
    err_console.print(
        f"[red]No pricing entry found for model:[/red] [bold]{model}[/bold]"
    )
    all_models = sorted({m for _, m in registry})
    _print_suggestions(model, all_models)


def _print_suggestions(model: str, all_models: list[str]) -> None:
    suggestions = difflib.get_close_matches(model, all_models, n=5, cutoff=0.4)
    if suggestions:
        err_console.print("[yellow]Did you mean:[/yellow]")
        for s in suggestions:
            err_console.print(f"  • {s}")
    else:
        err_console.print(
            "[yellow]Registered models:[/yellow] " + ", ".join(all_models)
        )


def _run_validate_prices(prices_file: str) -> None:
    """Implement ``bench cost --validate-prices <path>``.

    Exits 0 if every entry parsed cleanly, 1 if any entries were skipped, 2
    on hard errors (file missing, malformed YAML).
    """
    try:
        from inferencebench_llm.pricing import validate_pricing_file
    except ImportError as exc:
        err_console.print(
            "[red]--validate-prices requires the inferencebench-llm plugin.[/red]"
        )
        raise typer.Exit(code=2) from exc

    path = Path(prices_file)
    try:
        stats = validate_pricing_file(path)
    except FileNotFoundError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=2) from exc
    except ValueError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    if stats.skipped == 0 and stats.valid > 0:
        console.print(
            f"[green]{stats.valid} entries valid, {stats.skipped} skipped[/green]"
        )
        return

    if stats.valid == 0:
        err_console.print(
            f"[red]{stats.valid} entries valid, {stats.skipped} skipped[/red]"
        )
    else:
        err_console.print(
            f"[yellow]{stats.valid} entries valid, {stats.skipped} skipped[/yellow]"
        )
    for err in stats.errors:
        err_console.print(f"  • {err}")
    raise typer.Exit(code=1)
