"""Tests for ``bench coverage`` — plugin-aware metric completeness."""

from __future__ import annotations

import json
from pathlib import Path

from _helpers import (  # type: ignore[import-not-found]
    make_envelope,
    write_envelope_json,
)
from typer.testing import CliRunner

from inferencebench.cli import app

# Wide console so the Rich table doesn't truncate and substring assertions
# remain stable across Rich versions.
runner = CliRunner(env={"COLUMNS": "240"})


# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #
def _full_llm_inference_metrics() -> dict[str, float | int | str | None]:
    """Every metric llm-inference's ``EXPECTED_METRICS`` lists, all present."""
    return {
        "throughput_tok_per_s": 2000.0,
        "ttft_p50_ms": 120.0,
        "ttft_p99_ms": 380.0,
        "tpot_p50_ms": 18.0,
        "tpot_p99_ms": 32.0,
        "total_p50_ms": 1800.0,
        "total_p99_ms": 2500.0,
        "ok_rate": 0.99,
        "compliance_rate": 0.95,
        "power_avg_w": 720.0,
        "power_peak_w": 800.0,
        "energy_joules_total": 12000.0,
        "joules_per_token": 1.8,
        "req_per_s_passing": 16.4,
        "req_per_s_all": 17.0,
    }


def _write(
    path: Path,
    *,
    suite_id: str,
    metrics: dict[str, float | int | str | None],
    run_id: str = "01934567-89ab-7000-8000-000000000001",
) -> Path:
    env = make_envelope(
        model_id="meta-llama/Llama-4-Maverick",
        metrics=metrics,
        run_id=run_id,
        suite_id=suite_id,
    )
    return write_envelope_json(path, env)


# --------------------------------------------------------------------------- #
# Tests                                                                       #
# --------------------------------------------------------------------------- #
def test_coverage_full_envelope_reports_100pct(tmp_path: Path) -> None:
    """An envelope with every expected metric → 100% coverage, exit 0."""
    _write(
        tmp_path / "full.json",
        suite_id="llm.inference.sharegpt-v3",
        metrics=_full_llm_inference_metrics(),
    )
    result = runner.invoke(app, ["coverage", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "100.0%" in result.output
    assert "full.json" in result.output


def test_coverage_missing_three_metrics_reports_correct_counts(
    tmp_path: Path,
) -> None:
    """Dropping 3 metrics → expected=15, found=12, missing-list shows 3 names."""
    metrics = _full_llm_inference_metrics()
    metrics.pop("power_avg_w")
    metrics.pop("power_peak_w")
    metrics.pop("energy_joules_total")
    _write(
        tmp_path / "partial.json",
        suite_id="llm.inference.sharegpt-v3",
        metrics=metrics,
    )
    # Below the default 0.8 threshold (12/15 = 80% exactly) only if we drop
    # one more — but at exactly 80% the gate should *not* fire. To make the
    # exit-1 assertion deterministic we use a higher threshold here.
    result = runner.invoke(
        app, ["coverage", str(tmp_path), "--threshold", "0.95"]
    )
    assert result.exit_code == 1, result.output
    assert "partial.json" in result.output
    assert "80.0%" in result.output
    # Each dropped metric appears in the "missing" column.
    assert "power_avg_w" in result.output
    assert "power_peak_w" in result.output
    assert "energy_joules_total" in result.output


def test_coverage_threshold_above_passes_with_default(tmp_path: Path) -> None:
    """Coverage exactly at the default 0.8 threshold should NOT exit 1."""
    metrics = _full_llm_inference_metrics()
    # Drop 3/15 → 12/15 = 80.0% — at the default 0.8 threshold (>= not >).
    metrics.pop("power_avg_w")
    metrics.pop("power_peak_w")
    metrics.pop("energy_joules_total")
    _write(
        tmp_path / "border.json",
        suite_id="llm.inference.sharegpt-v3",
        metrics=metrics,
    )
    result = runner.invoke(app, ["coverage", str(tmp_path)])
    assert result.exit_code == 0, result.output


def test_coverage_threshold_below_fails(tmp_path: Path) -> None:
    """Coverage below --threshold → exit 1."""
    # Drop nearly all metrics, leaving 2 of 15 → 13.3%.
    sparse: dict[str, float | int | str | None] = {
        "throughput_tok_per_s": 1500.0,
        "ok_rate": 0.95,
    }
    _write(
        tmp_path / "sparse.json",
        suite_id="llm.inference.sharegpt-v3",
        metrics=sparse,
    )
    result = runner.invoke(app, ["coverage", str(tmp_path), "--threshold", "0.5"])
    assert result.exit_code == 1, result.output


def test_coverage_json_emits_structured_form(tmp_path: Path) -> None:
    """``--json`` returns a JSON dict with per-envelope rows."""
    metrics = _full_llm_inference_metrics()
    metrics.pop("ttft_p99_ms")
    _write(
        tmp_path / "j.json",
        suite_id="llm.inference.sharegpt-v3",
        metrics=metrics,
    )
    result = runner.invoke(
        app, ["coverage", str(tmp_path), "--json", "--threshold", "0.0"]
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert "envelopes" in payload
    assert "threshold" in payload
    [row] = payload["envelopes"]
    assert row["filename"] == "j.json"
    assert row["suite"] == "llm.inference.sharegpt-v3"
    assert row["expected_count"] == 15
    assert row["found_count"] == 14
    assert "ttft_p99_ms" in row["missing"]
    assert 93.0 < row["coverage_pct"] < 94.0


def test_coverage_single_file_input(tmp_path: Path) -> None:
    """Passing a file path rather than a directory still works."""
    f = _write(
        tmp_path / "single.json",
        suite_id="llm.inference.sharegpt-v3",
        metrics=_full_llm_inference_metrics(),
    )
    result = runner.invoke(app, ["coverage", str(f)])
    assert result.exit_code == 0, result.output
    assert "single.json" in result.output


def test_coverage_sorts_rows_worst_first(tmp_path: Path) -> None:
    """Envelopes sort by coverage ascending — worst row prints first."""
    good = _full_llm_inference_metrics()
    bad: dict[str, float | int | str | None] = {
        "throughput_tok_per_s": 1500.0,
        "ok_rate": 0.9,
    }
    _write(
        tmp_path / "good.json",
        suite_id="llm.inference.sharegpt-v3",
        metrics=good,
        run_id="01934567-89ab-7000-8000-0000000000a1",
    )
    _write(
        tmp_path / "bad.json",
        suite_id="llm.inference.sharegpt-v3",
        metrics=bad,
        run_id="01934567-89ab-7000-8000-0000000000a2",
    )
    result = runner.invoke(
        app, ["coverage", str(tmp_path), "--threshold", "0.0"]
    )
    assert result.exit_code == 0, result.output
    bad_idx = result.output.find("bad.json")
    good_idx = result.output.find("good.json")
    assert bad_idx != -1
    assert good_idx != -1
    assert bad_idx < good_idx, "bad.json (lower coverage) should print first"


def test_coverage_unknown_suite_id_has_empty_expected(tmp_path: Path) -> None:
    """Envelopes whose suite has no matching plugin get expected=0 → 100%."""
    _write(
        tmp_path / "alien.json",
        suite_id="alien.suite.foo",
        metrics={"some_metric": 1.0},
    )
    result = runner.invoke(
        app, ["coverage", str(tmp_path), "--json", "--threshold", "0.0"]
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    [row] = payload["envelopes"]
    assert row["expected_count"] == 0
    assert row["found_count"] == 0
    assert row["coverage_pct"] == 100.0


def test_coverage_missing_path_errors(tmp_path: Path) -> None:
    """Nonexistent path → exit 2 (matches summary/diff convention)."""
    result = runner.invoke(app, ["coverage", str(tmp_path / "nope")])
    assert result.exit_code == 2, result.output


def test_coverage_skips_invalid_envelopes(tmp_path: Path) -> None:
    """A junk JSON file is silently skipped, not counted in rows."""
    _write(
        tmp_path / "real.json",
        suite_id="llm.inference.sharegpt-v3",
        metrics=_full_llm_inference_metrics(),
    )
    (tmp_path / "junk.json").write_text(json.dumps({"hello": "world"}))
    result = runner.invoke(
        app, ["coverage", str(tmp_path), "--json", "--threshold", "0.0"]
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    filenames = [r["filename"] for r in payload["envelopes"]]
    assert filenames == ["real.json"]


def test_coverage_voice_suite_uses_voice_expected_metrics(tmp_path: Path) -> None:
    """A voice.transcription envelope is scored against the voice plugin's metrics."""
    full_voice: dict[str, float | int | str | None] = {
        "wer_mean": 0.18,
        "wer_p50": 0.16,
        "wer_p95": 0.34,
        "ok_rate": 1.0,
        "n_samples": 24,
        "total_audio_duration_s": 720.0,
        "total_p50_ms": 1500.0,
        "audio_path_resolved_count": 24,
    }
    _write(
        tmp_path / "voice.json",
        suite_id="voice.transcription.long-form",
        metrics=full_voice,
    )
    result = runner.invoke(
        app, ["coverage", str(tmp_path), "--json", "--threshold", "0.0"]
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    [row] = payload["envelopes"]
    assert row["expected_count"] == 8
    assert row["found_count"] == 8
    assert row["coverage_pct"] == 100.0
