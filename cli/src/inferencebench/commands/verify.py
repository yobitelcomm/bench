"""``bench verify`` — verify a signed envelope's Sigstore signature + content hash."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from inferencebench.envelope import Envelope, verify_envelope

console = Console()
err_console = Console(stderr=True)


def verify(
    envelope_uri: Annotated[
        str,
        typer.Argument(
            help="Envelope URI: local path, hf://datasets/..., or https://...",
        ),
    ],
    dev_public_key: Annotated[
        Path | None,
        typer.Option(
            "--dev-public-key",
            help="Path to ed25519 public key for dev-signed envelopes.",
        ),
    ] = None,
) -> None:
    """Verify a signed envelope. Exits 0 on success, non-zero on failure."""
    envelope = _load_envelope(envelope_uri)
    result = verify_envelope(envelope, dev_public_key_path=dev_public_key)

    if result.ok:
        console.print(f"[bold green]OK[/bold green]  {envelope_uri}")
        console.print(f"  method:           {result.method}")
        console.print(f"  content_hash:     {envelope.content_hash()}")
        console.print(f"  suite:            {envelope.suite_id} v{envelope.suite_version}")
        console.print(f"  model:            {envelope.model.id}")
        console.print(f"  engine:           {envelope.engine.name} v{envelope.engine.version}")
        if result.rekor_log_index >= 0:
            console.print(f"  rekor_log_index:  {result.rekor_log_index}")
        raise typer.Exit(code=0)

    err_console.print(f"[bold red]FAIL[/bold red]  {envelope_uri}")
    err_console.print(f"  method:  {result.method}")
    err_console.print(f"  reason:  {result.reason}")
    raise typer.Exit(code=1)


def _load_envelope(uri: str) -> Envelope:
    """Load an envelope from a URI. Phase 1: local file paths only."""
    if uri.startswith(("hf://", "https://", "s3://")):
        err_console.print(
            f"[red]URI scheme not yet supported in v0.0.0:[/red] {uri.split('://')[0]}://"
        )
        err_console.print("Phase 1 supports local file paths only. Download the envelope first.")
        raise typer.Exit(code=2)

    path = Path(uri)
    if not path.exists():
        err_console.print(f"[red]Envelope not found:[/red] {path}")
        raise typer.Exit(code=2)

    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        err_console.print(f"[red]Invalid JSON in envelope:[/red] {exc}")
        raise typer.Exit(code=2) from exc

    try:
        return Envelope.model_validate(raw)
    except Exception as exc:
        err_console.print(f"[red]Envelope schema validation failed:[/red] {exc}")
        raise typer.Exit(code=2) from exc
