"""``bench cache`` — manage the local envelope fetch cache.

:mod:`inferencebench.commands.fetch` writes downloaded envelopes to a local
cache directory (``~/.cache/inferencebench/fetched/`` by default). This
command surface exists so users can inspect what they've fetched, drop stale
entries, and grab the cache path for shell scripts.

The cache root is overridable via the ``BENCH_CACHE_ROOT`` environment
variable so tests (and power users) can redirect it without monkeypatching
``Path.home()``.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from inferencebench.envelope import Envelope

app = typer.Typer(no_args_is_help=True)
console = Console()
err_console = Console(stderr=True)


def _cache_root() -> Path:
    """Return the resolved cache directory.

    Honours ``BENCH_CACHE_ROOT`` if set; otherwise falls back to the same
    default ``bench fetch`` writes to.
    """
    override = os.environ.get("BENCH_CACHE_ROOT")
    if override:
        return Path(override)
    return Path.home() / ".cache" / "inferencebench" / "fetched"


def _format_size(num_bytes: int) -> str:
    """Render a byte count as a short human-readable string."""
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024.0 or unit == "GB":
            if unit == "B":
                return f"{int(size)} {unit}"
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} GB"


def _format_age(seconds: float) -> str:
    """Render an age in seconds as a compact 'NdMh' string."""
    if seconds < 60:
        return f"{int(seconds)}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m"
    if seconds < 86400:
        return f"{int(seconds // 3600)}h"
    return f"{int(seconds // 86400)}d"


def _iter_cache_files(root: Path) -> list[Path]:
    """Return cache files sorted by mtime descending (newest first)."""
    if not root.exists() or not root.is_dir():
        return []
    files = [p for p in root.iterdir() if p.is_file()]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files


def _try_load_envelope(path: Path) -> Envelope | None:
    """Attempt to parse ``path`` as an envelope; return ``None`` on failure.

    The cache may contain malformed or partial downloads; rather than crashing
    ``bench cache list`` we silently skip envelope-derived columns for those.
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return Envelope.model_validate(raw)
    except Exception:
        return None


@app.command("list")
def cache_list() -> None:
    """List cached envelopes with size, age, content hash, and suite/model."""
    root = _cache_root()
    files = _iter_cache_files(root)
    if not files:
        console.print(f"[yellow]no entries[/yellow]  (cache root: {root})")
        return

    table = Table(title=f"Cached envelopes ({root})")
    table.add_column("file", style="cyan")
    table.add_column("size", justify="right")
    table.add_column("age", justify="right")
    table.add_column("content_hash", style="dim")
    table.add_column("suite", style="green")
    table.add_column("model", style="bold")

    now = time.time()
    for path in files:
        stat = path.stat()
        envelope = _try_load_envelope(path)
        if envelope is not None:
            content_hash = envelope.content_hash()[:12]
            suite = envelope.suite_id
            model = envelope.model.id
        else:
            content_hash = "[red]invalid[/red]"
            suite = "-"
            model = "-"
        table.add_row(
            path.name,
            _format_size(stat.st_size),
            _format_age(now - stat.st_mtime),
            content_hash,
            suite,
            model,
        )
    console.print(table)


@app.command("clear")
def cache_clear(
    older_than: Annotated[
        int,
        typer.Option(
            "--older-than",
            help=(
                "Only delete cache files older than N days. With no flag, deletes every cache file."
            ),
        ),
    ] = -1,
    yes: Annotated[
        bool,
        typer.Option(
            "--yes/--no-yes",
            help="Skip the confirmation prompt.",
        ),
    ] = False,
) -> None:
    """Delete cache files. Prompts for confirmation unless ``--yes`` is set.

    Pass ``--older-than 7`` to drop files older than 7 days; pass
    ``--older-than 0`` (and ``--yes``) to drop everything that exists.
    """
    root = _cache_root()
    files = _iter_cache_files(root)
    if not files:
        console.print(f"[yellow]no entries to clear[/yellow]  (cache root: {root})")
        return

    if older_than >= 0:
        cutoff = time.time() - (older_than * 86400)
        targets = [p for p in files if p.stat().st_mtime <= cutoff]
    else:
        targets = list(files)

    if not targets:
        console.print(
            f"[yellow]nothing matches[/yellow]  (older-than={older_than}d, cache root: {root})"
        )
        return

    if not yes:
        confirm_label = (
            f"Delete {len(targets)} cache file(s) from {root}?"
            if older_than < 0
            else (f"Delete {len(targets)} cache file(s) older than {older_than}d from {root}?")
        )
        if not typer.confirm(confirm_label):
            console.print("[yellow]cancelled[/yellow]")
            raise typer.Exit(code=1)

    removed = 0
    for path in targets:
        try:
            path.unlink()
            removed += 1
        except OSError as exc:
            err_console.print(f"[red]failed to remove {path}:[/red] {exc}")
    console.print(f"[green]removed[/green] {removed} cache file(s)")


@app.command("path")
def cache_path() -> None:
    """Print the resolved cache root path on a single line.

    Useful for shell scripts: ``rm -rf "$(bench cache path)"``.
    """
    console.print(str(_cache_root()))
