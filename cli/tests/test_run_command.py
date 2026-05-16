"""Tests for ``bench run`` (ticket 0025).

These exercise the CLI wiring — plugin lookup, --list, error messages on
missing plugin. End-to-end execution against a live engine is out of scope
here (covered by integration tests in ``plugins/llm-inference/tests/``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from typer.testing import CliRunner

from inferencebench.cli import app
from inferencebench.commands import run as run_module

if TYPE_CHECKING:
    import pytest

runner = CliRunner()


def test_run_with_no_plugins_prints_helpful_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With zero plugins discovered, ``bench run`` exits non-zero with a hint."""
    monkeypatch.setattr(run_module, "_entry_points", lambda: [])

    result = runner.invoke(
        app,
        ["run", "llm.inference", "--model", "foo"],
    )
    assert result.exit_code != 0
    combined = result.stdout + (result.stderr or "")
    assert "No plugin" in combined or "no plugin" in combined.lower()
    assert "llm.inference" in combined


def test_run_with_unknown_suite_lists_installed() -> None:
    """If at least one plugin is installed but the requested suite isn't, list what IS."""
    real_eps = run_module._entry_points()
    # The llm-inference plugin is part of this workspace so should be discoverable.
    assert any(ep.name == "llm.inference" for ep in real_eps), (
        "expected the llm.inference plugin to be installed in the workspace"
    )

    result = runner.invoke(
        app,
        ["run", "voice.realtime", "--model", "foo"],
    )
    assert result.exit_code != 0
    combined = result.stdout + (result.stderr or "")
    assert "voice.realtime" in combined
    # Should mention the installed plugin in the "Installed:" list.
    assert "llm.inference" in combined


def test_run_list_prints_benchmark_ids() -> None:
    """``bench run llm.inference --list`` prints available benchmark_ids and exits 0."""
    result = runner.invoke(app, ["run", "llm.inference", "--list"])
    assert result.exit_code == 0, result.stdout + (result.stderr or "")
    assert "llm.inference.sharegpt-v3" in result.stdout


def test_run_help_lists_new_flags() -> None:
    """``bench run --help`` exposes the new flags added in ticket 0025."""
    result = runner.invoke(app, ["run", "--help"])
    assert result.exit_code == 0
    for flag in ("--signing-mode", "--dev-key", "--strict", "--list", "--base-url"):
        assert flag in result.stdout, f"missing flag: {flag}"
