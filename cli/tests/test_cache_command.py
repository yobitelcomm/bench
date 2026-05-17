"""Tests for ``bench cache`` — local fetch-cache management.

The cache command reads ``BENCH_CACHE_ROOT`` for an override so tests can
point the cache root at a tmp directory instead of monkeypatching
``Path.home()``. Covers all four documented subcommands: ``list`` (empty +
populated), ``path``, ``clear --yes`` (delete all), and ``clear
--older-than 0 --yes`` (delete everything that exists, since age >= 0).
"""

from __future__ import annotations

from pathlib import Path

from _helpers import (  # type: ignore[import-not-found]
    make_envelope,
    write_envelope_json,
)
from typer.testing import CliRunner

from inferencebench.cli import app

runner = CliRunner(env={"COLUMNS": "240"})


def _populate(root: Path, *, n: int) -> list[Path]:
    """Write ``n`` distinct envelopes into ``root`` and return their paths."""
    root.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for i in range(n):
        env = make_envelope(
            model_id=f"meta-llama/Llama-3.1-{i + 1}B-Instruct",
            run_id=f"01934567-89ab-7000-8000-0000000000{i:02d}",
            metrics={"throughput_tok_per_s": 1000.0 + i},
        )
        path = write_envelope_json(root / f"entry-{i:02d}.json", env)
        paths.append(path)
    return paths


# --------------------------------------------------------------------------- #
# list                                                                        #
# --------------------------------------------------------------------------- #
def test_cache_list_empty(tmp_path: Path) -> None:
    """An empty cache prints a friendly 'no entries' message."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    result = runner.invoke(
        app,
        ["cache", "list"],
        env={"BENCH_CACHE_ROOT": str(cache_dir), "COLUMNS": "240"},
    )
    assert result.exit_code == 0, result.stdout + (result.stderr or "")
    assert "no entries" in result.stdout.lower()


def test_cache_list_populated_shows_rows(tmp_path: Path) -> None:
    """Two envelopes → two rows; the envelope's model id appears in the table."""
    cache_dir = tmp_path / "cache"
    _populate(cache_dir, n=2)
    result = runner.invoke(
        app,
        ["cache", "list"],
        env={"BENCH_CACHE_ROOT": str(cache_dir), "COLUMNS": "240"},
    )
    assert result.exit_code == 0, result.stdout + (result.stderr or "")
    assert "entry-00.json" in result.stdout
    assert "entry-01.json" in result.stdout
    # Model id from at least one of the envelopes shows up in the rendered table.
    assert "Llama-3.1-1B-Instruct" in result.stdout
    assert "Llama-3.1-2B-Instruct" in result.stdout


# --------------------------------------------------------------------------- #
# path                                                                        #
# --------------------------------------------------------------------------- #
def test_cache_path_prints_resolved_path(tmp_path: Path) -> None:
    """``cache path`` prints a single line equal to the override env var value."""
    cache_dir = tmp_path / "some" / "cache" / "dir"
    result = runner.invoke(
        app,
        ["cache", "path"],
        env={"BENCH_CACHE_ROOT": str(cache_dir), "COLUMNS": "240"},
    )
    assert result.exit_code == 0, result.stdout + (result.stderr or "")
    # Rich may wrap long paths; assert the directory's filename is on the first line.
    lines = [ln for ln in result.stdout.splitlines() if ln.strip()]
    assert len(lines) == 1
    assert str(cache_dir) == lines[0].strip()


# --------------------------------------------------------------------------- #
# clear                                                                       #
# --------------------------------------------------------------------------- #
def test_cache_clear_yes_removes_all(tmp_path: Path) -> None:
    """``cache clear --yes`` with no age filter removes every entry."""
    cache_dir = tmp_path / "cache"
    _populate(cache_dir, n=3)
    assert len(list(cache_dir.iterdir())) == 3

    result = runner.invoke(
        app,
        ["cache", "clear", "--yes"],
        env={"BENCH_CACHE_ROOT": str(cache_dir), "COLUMNS": "240"},
    )
    assert result.exit_code == 0, result.stdout + (result.stderr or "")
    assert "removed" in result.stdout.lower()
    assert list(cache_dir.iterdir()) == []


def test_cache_clear_older_than_zero_removes_all(tmp_path: Path) -> None:
    """``--older-than 0 --yes`` drops every existing file (anything ≥ 0 days old)."""
    cache_dir = tmp_path / "cache"
    _populate(cache_dir, n=2)

    result = runner.invoke(
        app,
        ["cache", "clear", "--older-than", "0", "--yes"],
        env={"BENCH_CACHE_ROOT": str(cache_dir), "COLUMNS": "240"},
    )
    assert result.exit_code == 0, result.stdout + (result.stderr or "")
    assert list(cache_dir.iterdir()) == []
