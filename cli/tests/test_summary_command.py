"""Tests for ``bench summary``."""

from __future__ import annotations

import json
from pathlib import Path

from _helpers import (  # type: ignore[import-not-found]
    make_envelope,
    write_envelope_json,
)
from typer.testing import CliRunner

from inferencebench.cli import app

# Wide console so Rich tables don't truncate model ids in output assertions.
runner = CliRunner(env={"COLUMNS": "240"})


def _three_envelopes_in_dir(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Write three llm.inference envelopes into ``tmp_path``."""
    e1 = make_envelope(
        model_id="meta-llama/Llama-4-Maverick",
        run_id="01934567-89ab-7000-8000-000000000001",
        metrics={
            "throughput_tok_per_s": 1842.1,
            "ttft_p50_ms": 120.0,
            "ttft_p99_ms": 421.0,
            "tpot_p50_ms": 18.0,
            "ok_rate": 0.99,
            "power_avg_w": 750.0,
            "joules_per_token": 1.8,
            "cost_usd_per_million_tokens": 0.45,
        },
    )
    e2 = make_envelope(
        model_id="mistralai/Mistral-Large",
        run_id="01934567-89ab-7000-8000-000000000002",
        metrics={
            "throughput_tok_per_s": 2200.0,
            "ttft_p50_ms": 100.0,
            "ttft_p99_ms": 310.0,
            "tpot_p50_ms": 15.0,
            "ok_rate": 0.995,
            "power_avg_w": 780.0,
            "joules_per_token": 2.1,
            "cost_usd_per_million_tokens": 0.62,
        },
    )
    e3 = make_envelope(
        model_id="openai/gpt-4o-clone",
        run_id="01934567-89ab-7000-8000-000000000003",
        metrics={
            "throughput_tok_per_s": 1200.0,
            "ttft_p50_ms": 200.0,
            "ttft_p99_ms": 580.0,
            "tpot_p50_ms": 25.0,
            "ok_rate": 0.97,
            "power_avg_w": 700.0,
            "joules_per_token": 3.4,
            "cost_usd_per_million_tokens": 1.10,
        },
    )
    return (
        write_envelope_json(tmp_path / "a.json", e1),
        write_envelope_json(tmp_path / "b.json", e2),
        write_envelope_json(tmp_path / "c.json", e3),
    )


def test_summary_directory_with_three_envelopes(tmp_path: Path) -> None:
    """All three model ids appear in stdout and the command exits 0."""
    _three_envelopes_in_dir(tmp_path)
    result = runner.invoke(app, ["summary", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "Llama-4-Maverick" in result.output
    assert "Mistral-Large" in result.output
    assert "gpt-4o-clone" in result.output
    assert "3 envelopes loaded" in result.output


def test_summary_empty_directory(tmp_path: Path) -> None:
    """Empty directory prints a graceful 0-envelope footer with exit 0."""
    empty = tmp_path / "empty"
    empty.mkdir()
    result = runner.invoke(app, ["summary", str(empty)])
    assert result.exit_code == 0, result.output
    assert "0 envelopes loaded" in result.output


def test_summary_non_envelope_json_is_skipped(tmp_path: Path) -> None:
    """A JSON file that isn't a valid envelope bumps the skipped count."""
    _three_envelopes_in_dir(tmp_path)
    (tmp_path / "junk.json").write_text(json.dumps({"hello": "world"}))
    result = runner.invoke(app, ["summary", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "3 envelopes loaded" in result.output
    assert "1 skipped" in result.output


def test_summary_json_flag_is_parseable(tmp_path: Path) -> None:
    """``--json`` emits a parseable dict keyed by suite_id with a skipped count."""
    _three_envelopes_in_dir(tmp_path)
    (tmp_path / "junk.json").write_text(json.dumps({"hello": "world"}))
    result = runner.invoke(app, ["summary", str(tmp_path), "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert set(payload.keys()) == {"suites", "skipped"}
    assert payload["skipped"] == 1
    assert "llm.inference" in payload["suites"]
    rows = payload["suites"]["llm.inference"]
    assert len(rows) == 3
    model_ids = {row["model_id"] for row in rows}
    assert model_ids == {
        "meta-llama/Llama-4-Maverick",
        "mistralai/Mistral-Large",
        "openai/gpt-4o-clone",
    }
    # Each row carries the canonical metric set.
    for row in rows:
        assert set(row["metrics"].keys()) == {
            "throughput_tok_per_s",
            "ttft_p50_ms",
            "ttft_p99_ms",
            "tpot_p50_ms",
            "ok_rate",
            "power_avg_w",
            "joules_per_token",
            "cost_usd_per_million_tokens",
        }
        assert len(row["run_id_short"]) == 8


def test_summary_missing_path_errors(tmp_path: Path) -> None:
    """A non-existent path exits 2 with a red error."""
    result = runner.invoke(app, ["summary", str(tmp_path / "does-not-exist")])
    assert result.exit_code == 2
