"""Tests for ``bench diff`` — per-metric delta between two envelopes."""

from __future__ import annotations

import json
import math
from pathlib import Path

from _helpers import (  # type: ignore[import-not-found]
    make_envelope,
    write_envelope_json,
    write_signed_envelope_json,
)
from typer.testing import CliRunner

from inferencebench.cli import app

# Use a wide console so the Rich table doesn't truncate and so output-substring
# assertions stay stable.
runner = CliRunner(env={"COLUMNS": "240"})


# --------------------------------------------------------------------------- #
# Builder helpers                                                             #
# --------------------------------------------------------------------------- #
def _write_pair(
    tmp_path: Path,
    baseline_metrics: dict[str, float | int | None],
    candidate_metrics: dict[str, float | int | None],
    *,
    baseline_model: str = "meta-llama/Llama-4-Maverick",
    candidate_model: str = "meta-llama/Llama-4-Maverick",
) -> tuple[Path, Path]:
    """Build + write two envelope JSON files and return their paths."""
    baseline = make_envelope(
        model_id=baseline_model,
        run_id="01934567-89ab-7000-8000-0000000000a1",
        metrics=baseline_metrics,
    )
    candidate = make_envelope(
        model_id=candidate_model,
        run_id="01934567-89ab-7000-8000-0000000000a2",
        metrics=candidate_metrics,
    )
    a = write_envelope_json(tmp_path / "baseline.json", baseline)
    b = write_envelope_json(tmp_path / "candidate.json", candidate)
    return a, b


# --------------------------------------------------------------------------- #
# Tests                                                                       #
# --------------------------------------------------------------------------- #
def test_diff_identical_envelopes_all_no_change(tmp_path: Path) -> None:
    """Two identical envelopes → every metric classified ``no_change``."""
    metrics = {
        "throughput_tok_per_s": 2000.0,
        "ttft_p99_ms": 350.0,
        "joules_per_token": 2.0,
    }
    a, b = _write_pair(tmp_path, metrics, dict(metrics))
    result = runner.invoke(app, ["diff", str(a), str(b), "--report", "json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    for row in payload["metrics"]:
        assert row["verdict"] == "no_change", row


def test_diff_throughput_improvement(tmp_path: Path) -> None:
    """+50% throughput → improvement, green ↑ in table output."""
    a, b = _write_pair(
        tmp_path,
        {"throughput_tok_per_s": 1000.0},
        {"throughput_tok_per_s": 1500.0},
    )
    result = runner.invoke(app, ["diff", str(a), str(b), "--report", "json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    [row] = [m for m in payload["metrics"] if m["name"] == "throughput_tok_per_s"]
    assert row["verdict"] == "improvement"
    assert math.isclose(row["delta_rel"], 0.5, rel_tol=1e-9)

    # Table output → green ↑ for the improvement row.
    table_result = runner.invoke(app, ["diff", str(a), str(b)])
    assert table_result.exit_code == 0, table_result.output
    assert "throughput_tok_per_s" in table_result.output
    # Either the literal markup is rendered to ANSI or stripped; in either
    # case the arrow + word "improvement" must be present.
    assert "↑" in table_result.output
    assert "improvement" in table_result.output


def test_diff_latency_regression(tmp_path: Path) -> None:
    """+50% ttft_p99_ms → regression (latency going up is bad)."""
    a, b = _write_pair(
        tmp_path,
        {"ttft_p99_ms": 200.0},
        {"ttft_p99_ms": 300.0},
    )
    result = runner.invoke(app, ["diff", str(a), str(b), "--report", "json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    [row] = [m for m in payload["metrics"] if m["name"] == "ttft_p99_ms"]
    assert row["verdict"] == "regression"
    assert math.isclose(row["delta_rel"], 0.5, rel_tol=1e-9)

    table_result = runner.invoke(app, ["diff", str(a), str(b)])
    assert table_result.exit_code == 0, table_result.output
    assert "↑" in table_result.output
    assert "regression" in table_result.output


def test_diff_default_tolerance_swallows_small_delta(tmp_path: Path) -> None:
    """A 1% delta with the default --tolerance 0.02 → no_change."""
    a, b = _write_pair(
        tmp_path,
        {"throughput_tok_per_s": 1000.0},
        {"throughput_tok_per_s": 1010.0},  # +1%
    )
    result = runner.invoke(app, ["diff", str(a), str(b), "--report", "json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    [row] = [m for m in payload["metrics"] if m["name"] == "throughput_tok_per_s"]
    assert row["verdict"] == "no_change"


def test_diff_wide_tolerance_swallows_five_percent(tmp_path: Path) -> None:
    """A 5% delta with --tolerance 0.10 → no_change."""
    a, b = _write_pair(
        tmp_path,
        {"ttft_p99_ms": 200.0},
        {"ttft_p99_ms": 210.0},  # +5%
    )
    result = runner.invoke(app, ["diff", str(a), str(b), "--tolerance", "0.10", "--report", "json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    [row] = [m for m in payload["metrics"] if m["name"] == "ttft_p99_ms"]
    assert row["verdict"] == "no_change"


def test_diff_json_emits_documented_structure(tmp_path: Path) -> None:
    """JSON report has the documented top-level + per-row keys."""
    a, b = _write_pair(
        tmp_path,
        {
            "throughput_tok_per_s": 1000.0,
            "ttft_p99_ms": 200.0,
            "joules_per_token": 2.0,
        },
        {
            "throughput_tok_per_s": 1500.0,  # improvement
            "ttft_p99_ms": 300.0,  # regression
            "joules_per_token": 2.0,  # no_change
        },
    )
    result = runner.invoke(app, ["diff", str(a), str(b), "--report", "json"])
    assert result.exit_code == 0, result.output

    payload = json.loads(result.output)
    assert set(payload.keys()) == {
        "baseline_path",
        "candidate_path",
        "context_match",
        "metrics",
    }
    assert payload["baseline_path"] == str(a)
    assert payload["candidate_path"] == str(b)
    assert payload["context_match"]["all_match"] is True

    metric_names = {m["name"] for m in payload["metrics"]}
    assert metric_names == {
        "throughput_tok_per_s",
        "ttft_p99_ms",
        "joules_per_token",
    }
    for row in payload["metrics"]:
        assert set(row.keys()) == {
            "name",
            "baseline",
            "candidate",
            "delta_abs",
            "delta_rel",
            "verdict",
        }
        assert row["verdict"] in {
            "improvement",
            "regression",
            "no_change",
            "unknown",
            "missing",
        }


def test_diff_strict_exits_one_on_regression(tmp_path: Path) -> None:
    """``--strict`` turns any regression into exit code 1."""
    a, b = _write_pair(
        tmp_path,
        {"ttft_p99_ms": 200.0},
        {"ttft_p99_ms": 400.0},  # +100% regression
    )
    result = runner.invoke(app, ["diff", str(a), str(b), "--strict"])
    assert result.exit_code == 1, result.output


def test_diff_no_strict_exits_zero_even_with_regression(tmp_path: Path) -> None:
    """Without --strict, regressions still exit 0 (report-only mode)."""
    a, b = _write_pair(
        tmp_path,
        {"ttft_p99_ms": 200.0},
        {"ttft_p99_ms": 400.0},
    )
    result = runner.invoke(app, ["diff", str(a), str(b)])
    assert result.exit_code == 0, result.output
    assert "regression" in result.output


def test_diff_different_models_emits_yellow_warning(tmp_path: Path) -> None:
    """Different model ids → yellow context-mismatch warning shown."""
    a, b = _write_pair(
        tmp_path,
        {"throughput_tok_per_s": 1000.0},
        {"throughput_tok_per_s": 1500.0},
        baseline_model="meta-llama/Llama-4-Maverick",
        candidate_model="mistralai/Mistral-Large",
    )
    result = runner.invoke(app, ["diff", str(a), str(b)])
    assert result.exit_code == 0, result.output
    # Both the warning prefix and the offending field name must appear.
    assert "warning" in result.output.lower()
    assert "model_id" in result.output


def test_diff_nan_metric_does_not_crash(tmp_path: Path) -> None:
    """NaN metric values are treated as missing → no_change, no crash."""
    a, b = _write_pair(
        tmp_path,
        {"throughput_tok_per_s": float("nan")},
        {"throughput_tok_per_s": float("nan")},
    )
    result = runner.invoke(app, ["diff", str(a), str(b), "--report", "json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    [row] = [m for m in payload["metrics"] if m["name"] == "throughput_tok_per_s"]
    assert row["verdict"] == "no_change"
    assert row["baseline"] is None
    assert row["candidate"] is None


def test_diff_missing_in_candidate_marked_missing(tmp_path: Path) -> None:
    """Metric present in baseline but absent in candidate → 'missing'."""
    a, b = _write_pair(
        tmp_path,
        {
            "throughput_tok_per_s": 1000.0,
            "ttft_p99_ms": 200.0,
        },
        {"throughput_tok_per_s": 1000.0},
    )
    result = runner.invoke(app, ["diff", str(a), str(b), "--report", "json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    [row] = [m for m in payload["metrics"] if m["name"] == "ttft_p99_ms"]
    assert row["verdict"] == "missing"


def test_diff_baseline_zero_shows_nan_rel(tmp_path: Path) -> None:
    """Baseline = 0 → delta_rel cannot be computed; row still survives."""
    a, b = _write_pair(
        tmp_path,
        {"throughput_tok_per_s": 0.0},
        {"throughput_tok_per_s": 100.0},
    )
    result = runner.invoke(app, ["diff", str(a), str(b), "--report", "json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    [row] = [m for m in payload["metrics"] if m["name"] == "throughput_tok_per_s"]
    assert row["delta_abs"] == 100.0
    assert row["delta_rel"] is None
    # We still know direction → improvement.
    assert row["verdict"] == "improvement"

    table_result = runner.invoke(app, ["diff", str(a), str(b)])
    assert table_result.exit_code == 0, table_result.output
    assert "n/a" in table_result.output


def test_diff_unknown_metric_classified_unknown(tmp_path: Path) -> None:
    """Metric not in the direction-policy lists → verdict 'unknown'."""
    a, b = _write_pair(
        tmp_path,
        {"weird_custom_metric": 1.0},
        {"weird_custom_metric": 2.0},
    )
    result = runner.invoke(app, ["diff", str(a), str(b), "--report", "json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    [row] = [m for m in payload["metrics"] if m["name"] == "weird_custom_metric"]
    assert row["verdict"] == "unknown"
    assert row["delta_abs"] == 1.0


def test_diff_missing_file_errors(tmp_path: Path) -> None:
    a, _ = _write_pair(tmp_path, {"throughput_tok_per_s": 1.0}, {"throughput_tok_per_s": 1.0})
    result = runner.invoke(app, ["diff", str(a), str(tmp_path / "nope.json")])
    assert result.exit_code != 0


def test_diff_verify_passes_on_dev_signed(tmp_path: Path, dev_keypair: tuple[Path, Path]) -> None:
    """--verify succeeds when both envelopes are dev-signed and intact."""
    priv, _ = dev_keypair
    baseline = make_envelope(
        model_id="signed-base",
        run_id="01934567-89ab-7000-8000-000000003333",
        metrics={"throughput_tok_per_s": 1000.0},
    )
    candidate = make_envelope(
        model_id="signed-base",
        run_id="01934567-89ab-7000-8000-000000004444",
        metrics={"throughput_tok_per_s": 1100.0},
    )
    a = write_signed_envelope_json(tmp_path / "a.json", baseline, dev_key=priv)
    b = write_signed_envelope_json(tmp_path / "b.json", candidate, dev_key=priv)
    result = runner.invoke(app, ["diff", str(a), str(b), "--verify"])
    assert result.exit_code == 0, result.output
