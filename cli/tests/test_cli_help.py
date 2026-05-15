"""Tests: ``bench --help`` and per-command ``--help`` invocations work.

These are the minimal smoke tests for the CLI skeleton (ticket 0003). They verify
that the Typer wiring is correct and every subcommand registers properly.
"""

from __future__ import annotations

from typer.testing import CliRunner

from inferencebench.cli import __version__, app

runner = CliRunner()


def test_version_flag() -> None:
    """`bench --version` shows version and exits 0."""
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_root_help() -> None:
    """`bench --help` lists all subcommands."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in ("run", "compare", "publish", "verify", "leaderboard", "doctor", "cost", "plugin"):
        assert cmd in result.stdout


def test_no_args_shows_help() -> None:
    """`bench` with no args shows help (Typer exits 2 by convention)."""
    result = runner.invoke(app, [])
    # no_args_is_help=True in our config -> exit code 2 (typer convention for help-on-no-args)
    assert result.exit_code in (0, 2)


def test_run_command_help() -> None:
    """`bench run --help` shows the run flags."""
    result = runner.invoke(app, ["run", "--help"])
    assert result.exit_code == 0
    for flag in ("--model", "--engine", "--hardware", "--quant", "--seed"):
        assert flag in result.stdout


def test_compare_command_help() -> None:
    result = runner.invoke(app, ["compare", "--help"])
    assert result.exit_code == 0
    assert "--report" in result.stdout


def test_publish_command_help() -> None:
    result = runner.invoke(app, ["publish", "--help"])
    assert result.exit_code == 0
    assert "--to" in result.stdout


def test_verify_command_help() -> None:
    result = runner.invoke(app, ["verify", "--help"])
    assert result.exit_code == 0
    assert "envelope" in result.stdout.lower()


def test_doctor_command_help() -> None:
    result = runner.invoke(app, ["doctor", "--help"])
    assert result.exit_code == 0
    assert "--strict" in result.stdout


def test_cost_command_help() -> None:
    result = runner.invoke(app, ["cost", "--help"])
    assert result.exit_code == 0
    assert "--providers" in result.stdout


def test_plugin_list_with_zero_plugins() -> None:
    """`bench plugin list` should show 'No plugins installed' in a clean environment.

    Note: if a plugin happens to be installed (e.g. inferencebench-llm during dev), this
    test still passes — it just verifies the command exits 0.
    """
    result = runner.invoke(app, ["plugin", "list"])
    assert result.exit_code == 0


def test_plugin_info_missing_plugin_errors() -> None:
    """`bench plugin info <nonexistent>` exits non-zero with a helpful message."""
    result = runner.invoke(app, ["plugin", "info", "definitely-not-a-plugin"])
    assert result.exit_code != 0


def test_run_stub_message() -> None:
    """`bench run --help` shows that the run command is wired up.

    We deliberately don't test the stub's exit code with positional args —
    that's Typer parsing detail that'll change when 0025 wires the real command.
    """
    result = runner.invoke(app, ["run", "--help"])
    assert result.exit_code == 0
    assert "Run a benchmark" in result.stdout or "suite" in result.stdout.lower()
