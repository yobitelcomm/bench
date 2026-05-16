"""``bench cost`` — compare model cost across providers.

Phase 1 implementation (ticket 0027). Reads the in-process pricing registry
shipped with ``inferencebench-llm`` and renders a per-provider price table for
one model, with a configurable input/output blend.
"""

from __future__ import annotations

import difflib
from typing import TYPE_CHECKING, Annotated

import typer
from rich.console import Console
from rich.table import Table

if TYPE_CHECKING:
    from inferencebench_llm.pricing import ModelPricing

console = Console()
err_console = Console(stderr=True)


def cost(
    model: Annotated[str, typer.Argument(help="Model id (e.g. llama-4-maverick).")],
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
) -> None:
    """Compare model cost across providers using the bundled pricing registry."""
    if suite:
        # Informational only; reserved for Phase 2 suite-aware pricing.
        pass

    try:
        from inferencebench_llm.pricing import all_providers, lookup
    except ImportError as exc:
        err_console.print(
            "[red]bench cost requires the inferencebench-llm plugin "
            "to be installed.[/red]"
        )
        err_console.print("  pip install inferencebench-llm")
        raise typer.Exit(code=2) from exc

    requested = _parse_providers(providers)
    candidate_providers = requested if requested else all_providers()

    entries: list[ModelPricing] = []
    for provider in candidate_providers:
        priced = lookup(provider, model)
        if priced is not None:
            entries.append(priced)

    if not entries:
        _emit_not_found(model)
        raise typer.Exit(code=1)

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


def _emit_not_found(model: str) -> None:
    """Render a red error + similar-model suggestions for an unknown model."""
    from inferencebench_llm.pricing import all_providers, models_for

    err_console.print(
        f"[red]No pricing entry found for model:[/red] [bold]{model}[/bold]"
    )

    all_models = sorted(
        {m for provider in all_providers() for m in models_for(provider)}
    )
    suggestions = difflib.get_close_matches(model, all_models, n=5, cutoff=0.4)
    if suggestions:
        err_console.print("[yellow]Did you mean:[/yellow]")
        for s in suggestions:
            err_console.print(f"  • {s}")
    else:
        err_console.print(
            "[yellow]Registered models:[/yellow] " + ", ".join(all_models)
        )
