"""Tests for ``bench compare`` (ticket 0026)."""

from __future__ import annotations

import json
from pathlib import Path

from _helpers import (  # type: ignore[import-not-found]
    make_envelope,
    write_envelope_json,
    write_signed_envelope_json,
)
from typer.testing import CliRunner

from inferencebench.cli import app

# Use a wide console in tests so Rich tables don't truncate model ids when
# we assert against ``result.output``.
runner = CliRunner(env={"COLUMNS": "240"})


# --------------------------------------------------------------------------- #
# Builder helpers                                                             #
# --------------------------------------------------------------------------- #
def _three_envelopes_on_disk(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Three llm.inference envelopes with varied throughput / latency / cost."""
    e1 = make_envelope(
        model_id="meta-llama/Llama-4-Maverick",
        run_id="01934567-89ab-7000-8000-000000000001",
        metrics={
            "throughput_tok_per_s": 1842.1,
            "ttft_p99_ms": 421.0,
            "goodput_at_slo": 18.2,
            "cost_usd_per_million_tokens": 0.45,
            "joules_per_token": 1.8,
        },
    )
    e2 = make_envelope(
        model_id="mistralai/Mistral-Large",
        run_id="01934567-89ab-7000-8000-000000000002",
        metrics={
            "throughput_tok_per_s": 2200.0,
            "ttft_p99_ms": 310.0,
            "goodput_at_slo": 22.0,
            "cost_usd_per_million_tokens": 0.62,
            "joules_per_token": 2.1,
        },
    )
    e3 = make_envelope(
        model_id="openai/gpt-4o-clone",
        run_id="01934567-89ab-7000-8000-000000000003",
        metrics={
            "throughput_tok_per_s": 1200.0,
            "ttft_p99_ms": 580.0,
            "goodput_at_slo": 11.0,
            "cost_usd_per_million_tokens": 1.10,
            "joules_per_token": 3.4,
        },
    )
    return (
        write_envelope_json(tmp_path / "a.json", e1),
        write_envelope_json(tmp_path / "b.json", e2),
        write_envelope_json(tmp_path / "c.json", e3),
    )


# --------------------------------------------------------------------------- #
# Tests                                                                       #
# --------------------------------------------------------------------------- #
def test_compare_table_default(tmp_path: Path) -> None:
    """``bench compare a b`` exits 0 and prints both model ids in a table."""
    a, b, _ = _three_envelopes_on_disk(tmp_path)
    result = runner.invoke(app, ["compare", str(a), str(b)])
    assert result.exit_code == 0, result.output
    assert "Llama-4-Maverick" in result.output
    assert "Mistral-Large" in result.output
    assert "Pareto" in result.output


def test_compare_with_three_envelopes(tmp_path: Path) -> None:
    a, b, c = _three_envelopes_on_disk(tmp_path)
    result = runner.invoke(app, ["compare", str(a), str(b), str(c)])
    assert result.exit_code == 0, result.output
    assert "Llama-4-Maverick" in result.output
    assert "Mistral-Large" in result.output
    assert "gpt-4o-clone" in result.output


def test_compare_json_report_is_parseable(tmp_path: Path) -> None:
    a, b, c = _three_envelopes_on_disk(tmp_path)
    result = runner.invoke(
        app, ["compare", str(a), str(b), str(c), "--report", "json"]
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert set(payload.keys()) == {"runs", "pareto"}
    assert len(payload["runs"]) == 3
    # Pareto block contains the three known metric-pair labels.
    assert set(payload["pareto"].keys()) == {
        "quality_vs_cost",
        "throughput_vs_latency",
        "throughput_vs_energy",
    }
    # Each run has a per-label pareto map alongside its metrics.
    for run in payload["runs"]:
        assert set(run["pareto"].keys()) == {
            "quality_vs_cost",
            "throughput_vs_latency",
            "throughput_vs_energy",
        }
        assert run["model_id"]
        assert run["suite_id"] == "llm.inference"


def test_compare_requires_at_least_two_paths(tmp_path: Path) -> None:
    a, _, _ = _three_envelopes_on_disk(tmp_path)
    result = runner.invoke(app, ["compare", str(a)])
    assert result.exit_code != 0


def test_compare_zero_args_errors() -> None:
    result = runner.invoke(app, ["compare"])
    # Typer treats missing required positional as exit 2.
    assert result.exit_code != 0


def test_compare_missing_file_errors(tmp_path: Path) -> None:
    a, _, _ = _three_envelopes_on_disk(tmp_path)
    result = runner.invoke(
        app, ["compare", str(a), str(tmp_path / "nope.json")]
    )
    assert result.exit_code != 0


def test_compare_pareto_only_filters_rows(tmp_path: Path) -> None:
    """The dominated envelope ``c`` must drop out of a Pareto-only report."""
    a, b, c = _three_envelopes_on_disk(tmp_path)
    result = runner.invoke(
        app, ["compare", str(a), str(b), str(c), "--report", "pareto"]
    )
    assert result.exit_code == 0, result.output
    # ``c`` (gpt-4o-clone) is dominated on every axis by both ``a`` and ``b``
    # so it must not appear in pareto-only output.
    assert "gpt-4o-clone" not in result.output
    # ``b`` dominates ``a`` on throughput/latency/quality, so at least it survives.
    assert "Mistral-Large" in result.output


def test_compare_missing_metrics_does_not_crash(tmp_path: Path) -> None:
    """An envelope without the canonical metrics shows '-' but doesn't crash."""
    e1 = make_envelope(
        model_id="model-a",
        run_id="01934567-89ab-7000-8000-00000000aaaa",
        metrics={"throughput_tok_per_s": 1000.0},
    )
    e2 = make_envelope(
        model_id="model-b",
        run_id="01934567-89ab-7000-8000-00000000bbbb",
        metrics={"unrelated_metric": 1.0},
    )
    a = write_envelope_json(tmp_path / "a.json", e1)
    b = write_envelope_json(tmp_path / "b.json", e2)

    result = runner.invoke(app, ["compare", str(a), str(b)])
    assert result.exit_code == 0, result.output
    assert "model-a" in result.output
    assert "model-b" in result.output


def test_compare_verify_passes_on_dev_signed(
    tmp_path: Path, dev_keypair: tuple[Path, Path]
) -> None:
    """--verify succeeds when both envelopes are dev-signed and intact."""
    priv, _ = dev_keypair
    e1 = make_envelope(
        model_id="signed-a",
        run_id="01934567-89ab-7000-8000-000000001111",
        metrics={
            "throughput_tok_per_s": 1500.0,
            "ttft_p99_ms": 400.0,
        },
    )
    e2 = make_envelope(
        model_id="signed-b",
        run_id="01934567-89ab-7000-8000-000000002222",
        metrics={
            "throughput_tok_per_s": 1800.0,
            "ttft_p99_ms": 350.0,
        },
    )
    a = write_signed_envelope_json(tmp_path / "a.json", e1, dev_key=priv)
    b = write_signed_envelope_json(tmp_path / "b.json", e2, dev_key=priv)

    result = runner.invoke(app, ["compare", str(a), str(b), "--verify"])
    assert result.exit_code == 0, result.output


def test_compare_verify_fails_on_unsigned(tmp_path: Path) -> None:
    """--verify exits 1 if any envelope lacks a signature."""
    a, b, _ = _three_envelopes_on_disk(tmp_path)
    result = runner.invoke(app, ["compare", str(a), str(b), "--verify"])
    assert result.exit_code == 1
