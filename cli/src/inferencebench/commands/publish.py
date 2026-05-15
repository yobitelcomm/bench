"""``bench publish`` — publish a signed envelope to HF Hub or local mirror.

Phase 1 stub. Real implementation lands in ticket 0030.
"""

from __future__ import annotations

from typing import Annotated

import typer
from rich.console import Console

err_console = Console(stderr=True)


def publish(
    run_id: Annotated[str, typer.Argument(help="Run ID or envelope path to publish.")],
    to: Annotated[
        str, typer.Option("--to", help="Target: hf (Hugging Face Hub), local, studio.")
    ] = "hf",
    workspace: Annotated[str, typer.Option("--workspace", help="Workspace (Studio only).")] = "",
    tag: Annotated[str, typer.Option("--tag", help="Optional tag for this publish.")] = "",
) -> None:
    """Publish a signed envelope.

    Phase 1 stub — ticket 0030 will wire to integrations/hf-publisher.
    """
    err_console.print(
        f"[yellow][stub][/yellow] bench publish [bold]{run_id}[/bold] "
        f"--to {to} --workspace {workspace or '<none>'} --tag {tag or '<none>'} — "
        "not yet implemented in v0.0.0 (ticket 0030)."
    )
