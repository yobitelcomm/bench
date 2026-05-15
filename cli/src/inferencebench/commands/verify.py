"""``bench verify`` — verify a signed envelope's Sigstore signature + content hash.

Phase 1 stub. Real implementation lands in ticket 0031 (depends on ticket 0005).
"""

from __future__ import annotations

from typing import Annotated

import typer
from rich.console import Console

app = typer.Typer(no_args_is_help=True)
err_console = Console(stderr=True)


@app.callback(invoke_without_command=True)
def verify(
    envelope_uri: Annotated[
        str,
        typer.Argument(
            help="Envelope URI: local path, hf://datasets/..., or https://...",
        ),
    ],
) -> None:
    """Verify a signed envelope.

    Phase 1 stub — ticket 0031 will wire to envelope.verify() + Sigstore + Rekor.
    """
    err_console.print(
        f"[yellow][stub][/yellow] bench verify [bold]{envelope_uri}[/bold] — "
        "not yet implemented in v0.0.0 (ticket 0031)."
    )
