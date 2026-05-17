"""``bench audit`` — verify every envelope in a directory and summarise.

Run this against a published corpus before trusting it for downstream
decisions. For each envelope:

- Parse the JSON, fail loudly on schema errors.
- Recompute the canonical ``content_hash`` and check it matches what would
  go into the signature.
- Verify the signature (dev-key or Sigstore keyless, dispatched on the
  ``signature.method`` field).
- Confirm the hardware fingerprint is non-trivial (not the placeholder
  ``0000...`` sha used in tests).

The output is a Rich Table sorted with failures first. Exit code is 0 when
every envelope passes, 1 when any fails.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from inferencebench.envelope import Envelope, verify_envelope

console = Console()
err_console = Console(stderr=True)


def audit(
    path: Annotated[
        Path,
        typer.Argument(
            help="Directory or single envelope file to audit.",
            exists=True,
        ),
    ],
    dev_public_key: Annotated[
        Path | None,
        typer.Option(
            "--dev-public-key",
            help="Path to ed25519 public key for dev-signed envelopes.",
        ),
    ] = None,
    strict: Annotated[
        bool,
        typer.Option(
            "--strict/--no-strict",
            help="Exit non-zero if any envelope fails any check.",
        ),
    ] = True,
    report: Annotated[
        str,
        typer.Option(
            "--report",
            help="Output format: table or json.",
        ),
    ] = "table",
) -> None:
    """Audit every envelope under PATH for content-hash + signature validity."""
    targets = _collect_targets(path)
    if not targets:
        err_console.print(f"[yellow]No JSON files found under {path}[/yellow]")
        raise typer.Exit(code=0)

    rows: list[dict[str, Any]] = []
    for target in targets:
        rows.append(_audit_one(target, dev_public_key))

    if report == "json":
        _print_json(rows)
    else:
        _print_table(rows, path)

    n_fail = sum(1 for r in rows if not r["ok"])
    if strict and n_fail > 0:
        raise typer.Exit(code=1)
    raise typer.Exit(code=0)


def _collect_targets(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(p for p in path.rglob("*.json") if not p.name.startswith("samples-"))


def _audit_one(target: Path, dev_pub: Path | None) -> dict[str, Any]:
    row: dict[str, Any] = {
        "path": str(target),
        "model_id": "-",
        "method": "-",
        "ok": False,
        "reason": "",
        "content_hash_short": "",
    }
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        row["reason"] = f"json: {exc}"
        return row

    try:
        envelope = Envelope.model_validate(raw)
    except Exception as exc:
        row["reason"] = f"schema: {exc}"
        return row

    row["model_id"] = envelope.model.id
    row["content_hash_short"] = envelope.content_hash()[:12]
    if envelope.signature is None:
        row["method"] = "unsigned"
        row["reason"] = "no signature"
        return row

    row["method"] = envelope.signature.method
    fp = envelope.hardware_fingerprint.fingerprint_sha256
    if fp == "0" * 64:
        row["reason"] = "placeholder hardware_fingerprint"
        return row

    try:
        result = verify_envelope(envelope, dev_public_key_path=dev_pub)
    except Exception as exc:
        row["reason"] = f"verify: {exc}"
        return row

    if not result.ok:
        row["reason"] = result.reason or "signature mismatch"
        return row

    row["ok"] = True
    return row


def _print_table(rows: list[dict[str, Any]], path: Path) -> None:
    table = Table(title=f"Audit of {path}")
    table.add_column("status", justify="center")
    table.add_column("envelope", style="dim")
    table.add_column("model")
    table.add_column("method")
    table.add_column("content_hash")
    table.add_column("reason", style="red")
    rows_sorted = sorted(rows, key=lambda r: (r["ok"], r["path"]))
    for r in rows_sorted:
        marker = "[bold green]✓[/]" if r["ok"] else "[bold red]✗[/]"
        table.add_row(
            marker,
            Path(r["path"]).name,
            r["model_id"][:32],
            r["method"],
            r["content_hash_short"],
            "" if r["ok"] else r["reason"][:60],
        )
    console.print(table)
    n_ok = sum(1 for r in rows if r["ok"])
    console.print(
        f"[bold]{n_ok}[/bold] / {len(rows)} envelopes verified "
        f"({len(rows) - n_ok} failed)"
    )


def _print_json(rows: list[dict[str, Any]]) -> None:
    payload = {
        "schema": "inferencebench.audit.v1",
        "n_total": len(rows),
        "n_ok": sum(1 for r in rows if r["ok"]),
        "rows": rows,
    }
    console.print_json(data=payload)
