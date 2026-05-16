"""``bench diff`` must tolerate string-valued metrics (e.g. ``cost_source``).

When the envelope's metrics dict carries a qualitative tag like
``cost_source = "registry:groq"``, the diff command should not attempt any
numeric arithmetic on it — the row either drops out cleanly or surfaces as
``unknown`` / ``no_change``, never as a crash.
"""

from __future__ import annotations

import json
from pathlib import Path

from _helpers import make_envelope, write_envelope_json  # type: ignore[import-not-found]
from typer.testing import CliRunner

from inferencebench.cli import app

runner = CliRunner(env={"COLUMNS": "240"})


def _pair_with_cost_source(
    tmp_path: Path,
    baseline_source: str,
    candidate_source: str,
) -> tuple[Path, Path]:
    """Write a baseline+candidate pair that includes a string ``cost_source`` metric."""
    baseline = make_envelope(
        model_id="meta-llama/Llama-3.1-8B-Instruct",
        run_id="01934567-89ab-7000-8000-0000000000b1",
        metrics={
            "throughput_tok_per_s": 1000.0,
            "cost_usd_per_million_tokens": 0.06,
            "cost_source": baseline_source,
        },
    )
    candidate = make_envelope(
        model_id="meta-llama/Llama-3.1-8B-Instruct",
        run_id="01934567-89ab-7000-8000-0000000000b2",
        metrics={
            "throughput_tok_per_s": 1100.0,
            "cost_usd_per_million_tokens": 0.05,
            "cost_source": candidate_source,
        },
    )
    a = write_envelope_json(tmp_path / "baseline.json", baseline)
    b = write_envelope_json(tmp_path / "candidate.json", candidate)
    return a, b


def test_diff_with_string_cost_source_does_not_crash(tmp_path: Path) -> None:
    """Mismatched string cost_source values → diff still runs, no numeric ops."""
    a, b = _pair_with_cost_source(tmp_path, "registry:groq", "provider")
    result = runner.invoke(app, ["diff", str(a), str(b), "--report", "json"])
    assert result.exit_code == 0, result.output

    payload = json.loads(result.output)
    [cost_row] = [m for m in payload["metrics"] if m["name"] == "cost_source"]
    # Both string values collapse to None on the numeric track, so the verdict
    # is "no_change" (both sides absent on the numeric axis) and no deltas
    # were attempted.
    assert cost_row["baseline"] is None
    assert cost_row["candidate"] is None
    assert cost_row["delta_abs"] is None
    assert cost_row["delta_rel"] is None
    assert cost_row["verdict"] in {"no_change", "unknown"}


def test_diff_with_matching_string_cost_source(tmp_path: Path) -> None:
    """Same string on both sides → no_change verdict, table output is clean."""
    a, b = _pair_with_cost_source(tmp_path, "registry:groq", "registry:groq")
    result = runner.invoke(app, ["diff", str(a), str(b)])
    assert result.exit_code == 0, result.output
    assert "cost_source" in result.output


def test_diff_table_output_includes_string_metric(tmp_path: Path) -> None:
    """Table renderer must not raise when a metric value is a string."""
    a, b = _pair_with_cost_source(tmp_path, "provider", "registry:fireworks")
    result = runner.invoke(app, ["diff", str(a), str(b)])
    assert result.exit_code == 0, result.output
    # Row is rendered with "-" cells for the numeric columns, but the metric
    # name itself appears.
    assert "cost_source" in result.output
