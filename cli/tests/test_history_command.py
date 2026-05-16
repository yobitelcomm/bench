"""Tests for ``bench history`` — time-series view of one metric across runs."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from _helpers import (  # type: ignore[import-not-found]
    make_envelope,
    write_envelope_json,
)
from typer.testing import CliRunner

from inferencebench.cli import app
from inferencebench.envelope import Envelope

# Wide console so Rich tables don't wrap and break substring assertions.
runner = CliRunner(env={"COLUMNS": "240"})


def _with_timestamp(env: Envelope, when: datetime) -> Envelope:
    """Return a copy of ``env`` with ``timestamp`` overridden."""
    return env.model_copy(update={"timestamp": when})


def _three_envelopes(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Three envelopes spaced one day apart, with rising throughput."""
    e1 = _with_timestamp(
        make_envelope(
            model_id="meta-llama/Llama-4-Maverick",
            run_id="01934567-89ab-7000-8000-000000000001",
            metrics={"throughput_tok_per_s": 1000.0, "ttft_p50_ms": 200.0},
        ),
        datetime(2026, 5, 10, 10, 0, 0, tzinfo=UTC),
    )
    e2 = _with_timestamp(
        make_envelope(
            model_id="meta-llama/Llama-4-Maverick",
            run_id="01934567-89ab-7000-8000-000000000002",
            metrics={"throughput_tok_per_s": 1500.0, "ttft_p50_ms": 180.0},
        ),
        datetime(2026, 5, 11, 10, 0, 0, tzinfo=UTC),
    )
    e3 = _with_timestamp(
        make_envelope(
            model_id="meta-llama/Llama-4-Maverick",
            run_id="01934567-89ab-7000-8000-000000000003",
            metrics={"throughput_tok_per_s": 1800.0, "ttft_p50_ms": 150.0},
        ),
        datetime(2026, 5, 12, 10, 0, 0, tzinfo=UTC),
    )
    return (
        write_envelope_json(tmp_path / "a.json", e1),
        write_envelope_json(tmp_path / "b.json", e2),
        write_envelope_json(tmp_path / "c.json", e3),
    )


def test_history_three_envelopes_chronological(tmp_path: Path) -> None:
    """Three envelopes → three rows, ordered ascending by timestamp."""
    _three_envelopes(tmp_path)
    result = runner.invoke(app, ["history", str(tmp_path)])
    assert result.exit_code == 0, result.output
    out = result.output
    # The default metric is throughput_tok_per_s; all three values should appear.
    assert "1,000.0" in out or "1000.00" in out or "1000.0" in out
    assert "1,500.0" in out or "1500.0" in out
    assert "1,800.0" in out or "1800.0" in out
    # Earliest timestamp appears before the latest in the rendered output.
    idx_first = out.find("2026-05-10")
    idx_last = out.find("2026-05-12")
    assert idx_first != -1
    assert idx_last != -1
    assert idx_first < idx_last


def test_history_filter_model(tmp_path: Path) -> None:
    """``--filter-model`` drops envelopes whose model.id doesn't match."""
    _three_envelopes(tmp_path)
    other = _with_timestamp(
        make_envelope(
            model_id="mistralai/Mistral-Large",
            run_id="01934567-89ab-7000-8000-0000000000ff",
            metrics={"throughput_tok_per_s": 5000.0},
        ),
        datetime(2026, 5, 13, 10, 0, 0, tzinfo=UTC),
    )
    write_envelope_json(tmp_path / "other.json", other)

    result = runner.invoke(
        app,
        [
            "history",
            str(tmp_path),
            "--filter-model",
            "meta-llama/Llama-4-Maverick",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert len(payload["series"]) == 3
    assert all(p["model_id"] == "meta-llama/Llama-4-Maverick" for p in payload["series"])


def test_history_sparkline_present(tmp_path: Path) -> None:
    """A sparkline using Unicode block chars appears when >=2 values exist."""
    _three_envelopes(tmp_path)
    result = runner.invoke(app, ["history", str(tmp_path)])
    assert result.exit_code == 0, result.output
    spark_chars = "▁▂▃▄▅▆▇█"
    assert any(c in result.output for c in spark_chars)
    assert "min=" in result.output
    assert "max=" in result.output
    assert "median=" in result.output


def test_history_metric_switch(tmp_path: Path) -> None:
    """``--metric`` switches which key is tracked."""
    _three_envelopes(tmp_path)
    result = runner.invoke(
        app,
        ["history", str(tmp_path), "--metric", "ttft_p50_ms", "--json"],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["metric"] == "ttft_p50_ms"
    values = [p["value"] for p in payload["series"]]
    assert values == [200.0, 180.0, 150.0]


def test_history_json_structure(tmp_path: Path) -> None:
    """``--json`` emits {metric, filter, series, stats}."""
    _three_envelopes(tmp_path)
    result = runner.invoke(app, ["history", str(tmp_path), "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert set(payload.keys()) == {"metric", "filter", "series", "stats"}
    assert payload["metric"] == "throughput_tok_per_s"
    assert len(payload["series"]) == 3
    stats = payload["stats"]
    assert stats["min"] == 1000.0
    assert stats["max"] == 1800.0
    assert stats["median"] == 1500.0
    assert stats["first"] == 1000.0
    assert stats["last"] == 1800.0
    # Series entries have the expected keys.
    for point in payload["series"]:
        assert set(point.keys()) == {"timestamp", "model_id", "value", "run_id"}


def test_history_no_matches_after_filter(tmp_path: Path) -> None:
    """0 envelopes after filter → yellow ``no matches`` message, exit 0."""
    _three_envelopes(tmp_path)
    result = runner.invoke(
        app,
        [
            "history",
            str(tmp_path),
            "--filter-model",
            "nonexistent/Phantom-Model",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "no matches" in result.output
