"""``bench fixtures`` — fetch and manage real public dataset fixtures.

Every plugin ships TINY synthetic fixtures so its smoke benchmark runs offline.
This command surface lets users with Hugging Face access download the real
public datasets (FLORES-200, HumanEval, GSM8K, TruthfulQA, MS MARCO) and store
them under a known cache directory so plugins can opt in via
``spec.dataset.uri = "fixtures://<key>"`` (or, for plugins that key on
``spec.dataset.path``, ``fixtures://<key>`` in that field).

Subcommands:

- ``bench fixtures list`` — Rich table of every known fixture and whether it's
  cached locally.
- ``bench fixtures fetch <key>`` — download + convert + validate.
- ``bench fixtures path`` — print the resolved cache root.
- ``bench fixtures clear`` — drop all (or a specific) cached fixture.

The cache root is overridable via the ``BENCH_FIXTURES_ROOT`` environment
variable so tests (and power users) can redirect it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from inferencebench.commands._fixtures_adapters import ADAPTERS
from inferencebench.commands._fixtures_registry import FIXTURES, FixtureEntry

app = typer.Typer(no_args_is_help=True)
console = Console()
err_console = Console(stderr=True)

_INSTALL_HINT = (
    "The 'datasets' package is required to fetch fixtures. "
    "Install it with: pip install datasets"
)


def _cache_root() -> Path:
    """Return the resolved fixtures cache directory.

    Honours ``BENCH_FIXTURES_ROOT`` if set; otherwise falls back to
    ``~/.cache/inferencebench/fixtures/``.
    """
    override = os.environ.get("BENCH_FIXTURES_ROOT")
    if override:
        return Path(override)
    return Path.home() / ".cache" / "inferencebench" / "fixtures"


def _cache_path(key: str) -> Path:
    """Return the on-disk path for the cached fixture ``key``."""
    return _cache_root() / f"{key}.jsonl"


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


def _load_hf_rows(entry: FixtureEntry) -> list[dict[str, Any]]:
    """Pull rows for ``entry`` from Hugging Face. Raise ``typer.Exit`` if unavailable.

    Tries :func:`datasets.load_dataset` first. If the ``datasets`` package is
    not importable we exit with code 2 and a clear install hint — the
    ``huggingface_hub`` parquet fallback the task description mentions is
    best-effort and not implemented here because every fixture in the registry
    requires schema-aware row access (per-config splits, struct fields) that's
    awkward to reproduce by hand.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        err_console.print(f"[red]{_INSTALL_HINT}[/red]")
        raise typer.Exit(code=2) from None

    try:
        ds = load_dataset(
            entry.hf_dataset,
            entry.hf_config,
            split=entry.split,
            streaming=False,
        )
    except Exception as exc:  # pragma: no cover - network/HF failures
        err_console.print(f"[red]failed to load dataset:[/red] {exc}")
        raise typer.Exit(code=2) from exc

    return list(ds)


def _write_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write ``rows`` to ``path`` as jsonl using a tmp-rename atomic write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False))
            fp.write("\n")
    tmp.replace(path)


def _validate_jsonl(path: Path) -> int:
    """Confirm ``path`` is non-empty and every line parses. Return the row count.

    Raises:
        ValueError: If the file is empty or any line fails to parse.
    """
    count = 0
    with path.open("r", encoding="utf-8") as fp:
        for lineno, line in enumerate(fp, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                json.loads(stripped)
            except json.JSONDecodeError as exc:
                msg = f"line {lineno} of {path} is not valid JSON: {exc}"
                raise ValueError(msg) from exc
            count += 1
    if count == 0:
        msg = f"fixture {path} is empty after conversion"
        raise ValueError(msg)
    return count


@app.command("list")
def fixtures_list() -> None:
    """List every known fixture and whether it's cached locally."""
    root = _cache_root()
    table = Table(title=f"InferenceBench fixtures (cache root: {root})")
    table.add_column("key", style="cyan")
    table.add_column("dataset", style="dim")
    table.add_column("split")
    table.add_column("license", style="green")
    table.add_column("size", justify="right")
    table.add_column("cached", justify="center")

    for key in sorted(FIXTURES.keys()):
        entry = FIXTURES[key]
        path = _cache_path(key)
        if path.exists():
            cached_marker = "[green]yes[/green]"
            size_text = _format_size(path.stat().st_size)
        else:
            cached_marker = "[dim]no[/dim]"
            size_text = f"~{entry.size_estimate_mb} MB"
        dataset = entry.hf_dataset
        if entry.hf_config:
            dataset = f"{dataset}/{entry.hf_config}"
        table.add_row(
            key,
            dataset,
            entry.split,
            entry.license,
            size_text,
            cached_marker,
        )

    console.print(table)


@app.command("fetch")
def fixtures_fetch(
    key: Annotated[str, typer.Argument(help="Fixture key from `bench fixtures list`.")],
    force: Annotated[
        bool,
        typer.Option(
            "--force/--no-force",
            help="Re-download even if a cached copy already exists.",
        ),
    ] = False,
) -> None:
    """Download a fixture from Hugging Face into the local cache.

    The raw rows are routed through the registered adapter, then written
    atomically to ``<cache-root>/<key>.jsonl``. After write the file is
    validated (must be non-empty and every line must parse as JSON).
    """
    if key not in FIXTURES:
        err_console.print(f"[red]Unknown fixture key:[/red] {key}")
        err_console.print(f"Known keys: {', '.join(sorted(FIXTURES.keys()))}")
        raise typer.Exit(code=2)

    entry = FIXTURES[key]
    if entry.adapter not in ADAPTERS:  # pragma: no cover - defensive
        err_console.print(
            f"[red]Internal error:[/red] adapter '{entry.adapter}' not registered."
        )
        raise typer.Exit(code=2)

    out_path = _cache_path(key)
    if out_path.exists() and not force:
        console.print(
            f"[yellow]already cached[/yellow] at {out_path} "
            "(pass --force to re-download)"
        )
        return

    console.print(
        f"Fetching [bold]{key}[/bold] from "
        f"[dim]{entry.hf_dataset}"
        + (f"/{entry.hf_config}" if entry.hf_config else "")
        + f"[/dim] (split={entry.split}) ..."
    )

    raw_rows = _load_hf_rows(entry)
    adapter = ADAPTERS[entry.adapter]
    converted = list(adapter(raw_rows))

    if not converted:
        err_console.print(
            f"[red]Adapter '{entry.adapter}' produced 0 rows.[/red] "
            "The upstream dataset may have changed shape."
        )
        raise typer.Exit(code=2)

    _write_atomic(out_path, converted)

    try:
        row_count = _validate_jsonl(out_path)
    except ValueError as exc:
        err_console.print(f"[red]post-download validation failed:[/red] {exc}")
        # Remove the broken cache file so the next ``fetch`` retries cleanly.
        try:
            out_path.unlink()
        except OSError:
            pass
        raise typer.Exit(code=2) from exc

    size = _format_size(out_path.stat().st_size)
    console.print(
        f"[green]wrote[/green] {row_count} rows to {out_path} "
        f"(license: {entry.license}, size on disk: {size})"
    )


@app.command("path")
def fixtures_path() -> None:
    """Print the resolved cache root path on a single line."""
    console.print(str(_cache_root()))


@app.command("clear")
def fixtures_clear(
    key: Annotated[
        str | None,
        typer.Option(
            "--key",
            help="Clear only the named fixture. Omit to clear every cached fixture.",
        ),
    ] = None,
    yes: Annotated[
        bool,
        typer.Option(
            "--yes/--no-yes",
            help="Skip the confirmation prompt.",
        ),
    ] = False,
) -> None:
    """Delete cached fixtures.

    With ``--key <key>``, only that fixture is removed. Without it, every
    file inside the cache root is removed. Prompts for confirmation unless
    ``--yes`` is passed.
    """
    root = _cache_root()
    if not root.exists():
        console.print(f"[yellow]cache root does not exist:[/yellow] {root}")
        return

    if key is not None:
        if key not in FIXTURES:
            err_console.print(f"[red]Unknown fixture key:[/red] {key}")
            raise typer.Exit(code=2)
        target = _cache_path(key)
        if not target.exists():
            console.print(f"[yellow]not cached:[/yellow] {key}")
            return
        if not yes and not typer.confirm(f"Delete cached fixture {key} ({target})?"):
            console.print("[yellow]cancelled[/yellow]")
            raise typer.Exit(code=1)
        target.unlink()
        console.print(f"[green]removed[/green] {target}")
        return

    files = [p for p in root.iterdir() if p.is_file()]
    if not files:
        console.print(f"[yellow]no entries to clear[/yellow] (cache root: {root})")
        return
    if not yes and not typer.confirm(
        f"Delete {len(files)} cached fixture(s) from {root}?"
    ):
        console.print("[yellow]cancelled[/yellow]")
        raise typer.Exit(code=1)

    removed = 0
    for path in files:
        try:
            path.unlink()
            removed += 1
        except OSError as exc:
            err_console.print(f"[red]failed to remove {path}:[/red] {exc}")
    console.print(f"[green]removed[/green] {removed} cached fixture(s)")
