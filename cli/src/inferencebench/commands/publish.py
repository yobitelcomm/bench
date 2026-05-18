"""``bench publish`` — publish a signed envelope to HF Hub or save locally.

Wires the ``inferencebench-hf-publisher`` integration package.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from inferencebench.envelope import Envelope

console = Console()
err_console = Console(stderr=True)


def publish(
    run_id: Annotated[
        str,
        typer.Argument(help="Path to a signed envelope JSON file to publish."),
    ],
    to: Annotated[
        str,
        typer.Option(
            "--to",
            help="Target: hf (Hugging Face Hub), local (filesystem mirror).",
        ),
    ] = "hf",
    workspace: Annotated[
        str,
        typer.Option(
            "--workspace",
            help="Local mirror root (when --to local). Studio support is Phase 2+.",
        ),
    ] = "",
    tag: Annotated[
        str, typer.Option("--tag", help="Optional tag string recorded with the publish.")
    ] = "",
    org: Annotated[
        str,
        typer.Option(
            "--org",
            help="HF organisation to publish under. Defaults to Yobitel.",
        ),
    ] = "Yobitel",
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run/--no-dry-run",
            help="Plan the publish without touching the network or filesystem.",
        ),
    ] = False,
    raw_traces: Annotated[
        Path | None,
        typer.Option(
            "--raw-traces",
            help="Optional parquet file with raw request traces, uploaded alongside.",
        ),
    ] = None,
    update_model_card: Annotated[
        bool,
        typer.Option(
            "--update-model-card",
            help="Append a backlink entry to the source model card (best-effort).",
        ),
    ] = False,
) -> None:
    """Publish a signed envelope to HF Hub or a local mirror."""
    envelope_path = Path(run_id)
    if not envelope_path.exists():
        err_console.print(f"[red]Envelope not found:[/red] {envelope_path}")
        raise typer.Exit(code=2)

    try:
        envelope = Envelope.model_validate(json.loads(envelope_path.read_text("utf-8")))
    except Exception as exc:
        err_console.print(f"[red]Failed to load envelope:[/red] {exc}")
        raise typer.Exit(code=2) from exc

    if to == "hf":
        _publish_to_hf(
            envelope,
            org=org,
            dry_run=dry_run,
            raw_traces=raw_traces,
            update_model_card=update_model_card,
            tag=tag,
        )
        return
    if to == "local":
        _publish_to_local(envelope_path, envelope, workspace=workspace, tag=tag)
        return
    if to == "studio":
        err_console.print(
            "[yellow]Studio publishing is deferred to Phase 2+.[/yellow] Use --to hf or --to local."
        )
        raise typer.Exit(code=2)

    err_console.print(f"[red]Unknown --to target:[/red] {to}")
    raise typer.Exit(code=2)


def _publish_to_hf(
    envelope: Envelope,
    *,
    org: str,
    dry_run: bool,
    raw_traces: Path | None,
    update_model_card: bool,
    tag: str,
) -> None:
    try:
        from inferencebench_hf_publisher import (
            HfPublishError,
            HfRateLimitError,
            HfRepoCollisionError,
            publish_envelope_to_hf,
        )
    except ImportError as exc:
        err_console.print(
            "[red]inferencebench-hf-publisher is not installed.[/red] "
            "Install it: [bold]pip install inferencebench-hf-publisher[/bold]"
        )
        raise typer.Exit(code=2) from exc

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not dry_run and not hf_token:
        err_console.print(
            "[red]No HF token found.[/red] Set HF_TOKEN, run [bold]huggingface-cli login[/bold], "
            "or pass --dry-run to plan without uploading."
        )
        raise typer.Exit(code=2)

    try:
        result = publish_envelope_to_hf(
            envelope,
            hf_token=hf_token,
            raw_traces_path=raw_traces,
            update_model_card=update_model_card,
            org=org,
            dry_run=dry_run,
        )
    except HfRepoCollisionError as exc:
        err_console.print(f"[red]Repo collision:[/red] {exc}")
        err_console.print(
            "Regenerate the run_id and retry, or pass --org to publish under a different namespace."
        )
        raise typer.Exit(code=1) from exc
    except HfRateLimitError as exc:
        err_console.print(f"[red]HF Hub rate-limit:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    except HfPublishError as exc:
        err_console.print(f"[red]HF publish failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    prefix = "[yellow]DRY-RUN[/yellow] " if dry_run else "[bold green]OK[/bold green] "
    console.print(f"{prefix}published {envelope.suite_id} run to {result.url}")
    if tag:
        console.print(f"  tag:           {tag}")
    console.print(f"  repo_id:       {result.repo_id}")
    console.print(f"  files:         {len(result.files_uploaded)}")
    console.print(f"  verified:      {result.verified}")


def _publish_to_local(
    envelope_path: Path, envelope: Envelope, *, workspace: str, tag: str
) -> None:
    root = Path(workspace) if workspace else Path.cwd() / "bench-mirror"
    suite_slug = envelope.suite_id.replace(".", "-")
    target_dir = root / suite_slug
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{envelope.content_hash()[:12]}.json"
    shutil.copy2(envelope_path, target)
    _update_mirror_index(root, suite_slug, envelope, target, tag=tag)
    console.print(f"[bold green]OK[/bold green] mirrored to {target}")
    if tag:
        (target_dir / f"{target.stem}.tag").write_text(tag, encoding="utf-8")
        console.print(f"  tag:  {tag}")


def _update_mirror_index(
    root: Path, suite_slug: str, envelope: Envelope, target: Path, *, tag: str = ""
) -> None:
    """Maintain a self-describing ``index.json`` at the mirror root.

    Each entry: ``{suite_id, suite_slug, model_id, content_hash, path, signed,
    tag, timestamp}``. The index is append-style — we re-load it, add or
    update the entry for this target, sort by timestamp desc, and write back.
    Existing entries with the same ``path`` are replaced.
    """
    index_path = root / "index.json"
    entries: list[dict[str, str | int | bool]] = []
    if index_path.exists():
        try:
            payload = json.loads(index_path.read_text(encoding="utf-8"))
            existing = payload.get("entries") if isinstance(payload, dict) else None
            if isinstance(existing, list):
                entries = [e for e in existing if isinstance(e, dict)]
        except (OSError, ValueError):
            entries = []

    rel_path = str(target.relative_to(root))
    new_entry: dict[str, str | int | bool] = {
        "suite_id": str(envelope.suite_id),
        "suite_slug": suite_slug,
        "model_id": str(envelope.model.id),
        "engine": f"{envelope.engine.name} v{envelope.engine.version}",
        "content_hash": envelope.content_hash(),
        "path": rel_path,
        "signed": envelope.signature is not None,
        "tag": tag,
        "timestamp": envelope.timestamp.isoformat(),
    }
    entries = [e for e in entries if e.get("path") != rel_path]
    entries.append(new_entry)
    entries.sort(key=lambda e: str(e.get("timestamp", "")), reverse=True)

    index_payload = {
        "schema": "inferencebench.mirror.v1",
        "n_entries": len(entries),
        "entries": entries,
    }
    tmp = index_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(index_payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(index_path)
