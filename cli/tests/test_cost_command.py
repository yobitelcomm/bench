"""Tests for ``bench cost`` (ticket 0027)."""

from __future__ import annotations

from typer.testing import CliRunner

from inferencebench.cli import app

runner = CliRunner(env={"COLUMNS": "240"})


def test_cost_known_model_lists_providers() -> None:
    """A registered model resolves to at least one provider in the table."""
    result = runner.invoke(app, ["cost", "meta-llama/Llama-4-Maverick"])
    assert result.exit_code == 0, result.output
    # At least one of the registered providers must show up.
    assert any(p in result.output for p in ("together", "fireworks", "groq"))
    # Header rendered.
    assert "Provider" in result.output
    assert "Blended" in result.output


def test_cost_known_model_short_form() -> None:
    """OpenAI's ``gpt-4o`` resolves without a provider prefix."""
    result = runner.invoke(app, ["cost", "gpt-4o"])
    assert result.exit_code == 0, result.output
    assert "openai" in result.output


def test_cost_filtered_providers() -> None:
    """``--providers`` filters the registry to a subset."""
    result = runner.invoke(
        app,
        [
            "cost",
            "meta-llama/Llama-4-Maverick",
            "--providers",
            "together,fireworks",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "together" in result.output
    assert "fireworks" in result.output
    # groq was not requested.
    assert "groq" not in result.output


def test_cost_input_token_share_changes_blend() -> None:
    """A different ``--input-token-share`` runs without crashing."""
    result = runner.invoke(
        app,
        [
            "cost",
            "meta-llama/Llama-4-Maverick",
            "--input-token-share",
            "0.5",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "1:1" in result.output or "0.50" in result.output


def test_cost_unknown_model_errors_with_suggestions() -> None:
    """A made-up model exits 1 and suggests similar names if any."""
    result = runner.invoke(app, ["cost", "made-up-model-xyz"])
    assert result.exit_code == 1
    assert "No pricing entry" in result.output or "No pricing entry" in (
        result.stderr if hasattr(result, "stderr") else ""
    )


def test_cost_typo_near_real_name_suggests_match() -> None:
    """A close typo of a real model name should produce a 'Did you mean' line."""
    # 'gpt-4-o' is a single-character distance from 'gpt-4o'.
    result = runner.invoke(app, ["cost", "gpt-4-o"])
    assert result.exit_code == 1
    # Either a "Did you mean" hint or at least the registered list.
    combined = result.output
    assert "gpt-4o" in combined or "Registered models" in combined


def test_cost_suite_flag_is_accepted() -> None:
    """``--suite`` is currently informational but must not break the command."""
    result = runner.invoke(
        app,
        [
            "cost",
            "meta-llama/Llama-4-Maverick",
            "--suite",
            "intelligence-index",
        ],
    )
    assert result.exit_code == 0, result.output
