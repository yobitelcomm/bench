"""``bench cost`` — compare model cost across providers.

Phase 1 stub. Real implementation lands in ticket 0027.
"""

from __future__ import annotations

from typing import Annotated

import typer
from rich.console import Console

err_console = Console(stderr=True)


def cost(
    model: Annotated[str, typer.Argument(help="Model id (e.g. llama-4-maverick).")],
    suite: Annotated[
        str, typer.Option("--suite", help="Suite for the cost comparison.")
    ] = "intelligence-index",
    providers: Annotated[
        str,
        typer.Option(
            "--providers",
            help="Comma-separated provider list (together,fireworks,groq,...).",
        ),
    ] = "",
) -> None:
    """Compare model cost across providers.

    Phase 1 stub — ticket 0027 will wire to the pricing registry + suite results.
    """
    err_console.print(
        f"[yellow][stub][/yellow] bench cost [bold]{model}[/bold] "
        f"--suite {suite} --providers {providers or '<all>'} — "
        "not yet implemented in v0.0.0 (ticket 0027)."
    )
