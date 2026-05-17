"""Tests for ``bench tour`` — end-to-end install-validation walkthrough."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from inferencebench.cli import app

# Wide console so the summary table doesn't truncate step names we substring-check.
runner = CliRunner(env={"COLUMNS": "240"})


@pytest.fixture
def chdir_tmp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Change cwd to tmp_path so default ``--out`` resolves under it."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_tour_exits_0_on_fresh_tmp_path(chdir_tmp: Path) -> None:
    """A fresh tmp_path with default flags produces exit 0."""
    out = chdir_tmp / "tour-out"
    result = runner.invoke(app, ["tour", "--out", str(out)])
    assert result.exit_code == 0, result.output


def test_tour_produces_expected_outputs(chdir_tmp: Path) -> None:
    """The tour writes envelope, bundle, site/index.html, and markdown export."""
    out = chdir_tmp / "tour-out"
    result = runner.invoke(app, ["tour", "--out", str(out)])
    assert result.exit_code == 0, result.output

    assert (out / "tour-envelope.json").exists(), "tour-envelope.json missing"
    assert (out / "tour.bundle.zip").exists(), "tour.bundle.zip missing"
    assert (out / "site" / "index.html").exists(), "leaderboard site missing"
    assert (out / "tour-envelope.md").exists(), "markdown export missing"


def test_tour_summary_table_lists_ten_steps(chdir_tmp: Path) -> None:
    """The final summary lists every one of the 10 tour steps with a check mark."""
    out = chdir_tmp / "tour-out"
    result = runner.invoke(app, ["tour", "--out", str(out)])
    assert result.exit_code == 0, result.output

    # Every step name shows up in the summary table.
    for step_name in (
        "bench list",
        "bench plugin init",
        "generate dev keypair",
        "build + sign fake envelope",
        "bench verify",
        "bench summary",
        "bench export --format markdown",
        "bench bundle create",
        "bench leaderboard --build",
        "bench audit",
    ):
        assert step_name in result.output, f"step missing in summary: {step_name}"

    # The footer reports 10/10 passed.
    assert "10 / 10 steps passed" in result.output


def test_tour_creates_out_directory_if_absent(chdir_tmp: Path) -> None:
    """``--out`` directory is created on demand even if missing."""
    nested = chdir_tmp / "does" / "not" / "exist-yet"
    assert not nested.exists()
    result = runner.invoke(app, ["tour", "--out", str(nested)])
    assert result.exit_code == 0, result.output
    assert nested.exists()
    assert (nested / "tour-envelope.json").exists()
