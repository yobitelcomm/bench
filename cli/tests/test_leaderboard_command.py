"""Tests for ``bench leaderboard`` (ticket 0032)."""

from __future__ import annotations

from pathlib import Path

from _helpers import make_envelope, write_envelope_json  # type: ignore[import-not-found]
from typer.testing import CliRunner

from inferencebench.cli import app

runner = CliRunner(env={"COLUMNS": "240"})


def test_leaderboard_no_build_flag_directs_to_build_mode() -> None:
    result = runner.invoke(app, ["leaderboard"])
    assert result.exit_code != 0
    assert "--build" in result.output


def test_leaderboard_build_requires_envelopes_and_out() -> None:
    result = runner.invoke(app, ["leaderboard", "--build"])
    assert result.exit_code != 0


def test_leaderboard_build_renders_static_site(tmp_path: Path) -> None:
    envelopes_dir = tmp_path / "envelopes"
    envelopes_dir.mkdir()
    e1 = make_envelope(
        model_id="meta-llama/Llama-4-Maverick",
        run_id="01934567-89ab-7000-8000-000000000001",
        metrics={
            "throughput_tok_per_s": 1500.0,
            "ttft_p99_ms": 400.0,
            "cost_usd_per_million_tokens": 0.5,
        },
    )
    e2 = make_envelope(
        model_id="mistralai/Mistral-Large",
        run_id="01934567-89ab-7000-8000-000000000002",
        metrics={
            "throughput_tok_per_s": 1800.0,
            "ttft_p99_ms": 350.0,
            "cost_usd_per_million_tokens": 0.6,
        },
    )
    write_envelope_json(envelopes_dir / "a.json", e1)
    write_envelope_json(envelopes_dir / "b.json", e2)

    out_dir = tmp_path / "site"
    result = runner.invoke(
        app,
        [
            "leaderboard",
            "--build",
            "--envelopes",
            str(envelopes_dir),
            "--out",
            str(out_dir),
        ],
    )
    assert result.exit_code == 0, result.output
    assert (out_dir / "index.html").exists()
    assert "envelopes loaded" in result.output
