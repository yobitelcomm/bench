"""``bench compare`` — compare benchmark runs, render Pareto frontier.

Phase 1 stub. Real implementation lands in ticket 0026.
"""

from __future__ import annotations

from typing import Annotated

import typer
from rich.console import Console

app = typer.Typer(no_args_is_help=True)
err_console = Console(stderr=True)


@app.callback(invoke_without_command=True)
def compare(
    run_ids: Annotated[
        list[str], typer.Argument(help="One or more run IDs / envelope paths to compare.")
    ],
    report: Annotated[
        str,
        typer.Option(
            "--report",
            help="Report format: pareto, table, json.",
        ),
    ] = "pareto",
) -> None:
    """Compare two or more benchmark runs.

    Phase 1 stub — ticket 0026 will wire to envelope reader + Pareto renderer.
    """
    err_console.print(
        f"[yellow][stub][/yellow] bench compare {' '.join(run_ids)} "
        f"--report {report} — not yet implemented in v0.0.0 (ticket 0026)."
    )
