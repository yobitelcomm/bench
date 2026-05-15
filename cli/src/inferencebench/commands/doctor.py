"""``bench doctor`` — diagnose hardware health before benchmarking.

Phase 1 stub. Real implementation lands in ticket 0007.
"""

from __future__ import annotations

from typing import Annotated

import typer
from rich.console import Console

err_console = Console(stderr=True)


def doctor(
    strict: Annotated[
        bool,
        typer.Option(
            "--strict",
            help="Refuse if thermal throttling, ECC errors, or driver drift detected.",
        ),
    ] = False,
) -> None:
    """Run hardware diagnostic.

    Phase 1 stub — ticket 0007 will wire to harness/fingerprint + harness/telemetry.
    """
    err_console.print(
        f"[yellow][stub][/yellow] bench doctor --strict={strict} — "
        "not yet implemented in v0.0.0 (ticket 0007)."
    )
