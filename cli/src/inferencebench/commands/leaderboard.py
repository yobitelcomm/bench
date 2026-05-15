"""``bench leaderboard`` — browse public benchmark leaderboards.

Phase 1 stub. Real implementation lands in ticket 0032.
"""

from __future__ import annotations

from typing import Annotated

import typer
from rich.console import Console

err_console = Console(stderr=True)


def leaderboard(
    category: Annotated[
        str,
        typer.Argument(help="Category id (e.g. llm.inference). Omit to list categories."),
    ] = "",
) -> None:
    """Show the public leaderboard for a category.

    Phase 1 stub — ticket 0032 will fetch from yobitelcomm.github.io/bench.
    """
    if category:
        err_console.print(
            f"[yellow][stub][/yellow] bench leaderboard [bold]{category}[/bold] — "
            "not yet implemented in v0.0.0 (ticket 0032)."
        )
    else:
        err_console.print(
            "[yellow][stub][/yellow] bench leaderboard — list categories not yet "
            "implemented in v0.0.0 (ticket 0032)."
        )
