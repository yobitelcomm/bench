"""Tests for ``bench export`` — markdown / CSV / Slack envelope conversions."""

from __future__ import annotations

from pathlib import Path

from _helpers import (  # type: ignore[import-not-found]
    make_envelope,
    write_envelope_json,
)
from typer.testing import CliRunner

from inferencebench.cli import app

# Wide console so Rich doesn't truncate output we substring-match against.
runner = CliRunner(env={"COLUMNS": "240"})


# --------------------------------------------------------------------------- #
# Fixture builders                                                            #
# --------------------------------------------------------------------------- #
_LLAMA_METRICS: dict[str, float | int | str | None] = {
    "throughput_tok_per_s": 1384.24,
    "ttft_p50_ms": 41.69,
    "ttft_p99_ms": 64.71,
    "tpot_p50_ms": 10.94,
    "tpot_p99_ms": 13.50,
    "ok_rate": 1.0,
    "compliance_rate": 1.0,
    "power_avg_w": 972.8,
    "power_peak_w": 983.0,
    "joules_per_token": 0.70,
    "cost_usd_per_million_tokens": 0.42,
    "cost_source": "registry:groq",
}


def _write_llama_envelope(tmp_path: Path) -> Path:
    """Write a Llama-3.1-8B envelope JSON and return its path."""
    env = make_envelope(
        model_id="meta-llama/Llama-3.1-8B",
        metrics=dict(_LLAMA_METRICS),
    )
    return write_envelope_json(tmp_path / "llama.json", env)


# --------------------------------------------------------------------------- #
# Markdown                                                                    #
# --------------------------------------------------------------------------- #
def test_export_markdown_contains_header_and_table(tmp_path: Path) -> None:
    """Markdown output: header, model id, metric table header all present."""
    path = _write_llama_envelope(tmp_path)
    result = runner.invoke(app, ["export", str(path), "--format", "markdown"])
    assert result.exit_code == 0, result.output
    assert "## InferenceBench result" in result.output
    assert "meta-llama/Llama-3.1-8B" in result.output
    assert "| metric | value |" in result.output
    assert "|---|---|" in result.output
    # Spot-check a metric row.
    assert "throughput_tok_per_s" in result.output


def test_export_markdown_is_default_format(tmp_path: Path) -> None:
    """Markdown is the default when ``--format`` is omitted."""
    path = _write_llama_envelope(tmp_path)
    result = runner.invoke(app, ["export", str(path)])
    assert result.exit_code == 0, result.output
    assert "## InferenceBench result" in result.output


def test_export_markdown_renders_string_metric_verbatim(tmp_path: Path) -> None:
    """String-valued metrics (e.g. ``cost_source``) render without crashing."""
    env = make_envelope(
        model_id="meta-llama/Llama-3.1-8B",
        metrics={
            "throughput_tok_per_s": 1000.0,
            "cost_source": "registry:groq",
        },
    )
    path = write_envelope_json(tmp_path / "env.json", env)
    result = runner.invoke(app, ["export", str(path), "--format", "markdown"])
    assert result.exit_code == 0, result.output
    assert "cost_source" in result.output
    assert "registry:groq" in result.output


# --------------------------------------------------------------------------- #
# CSV                                                                         #
# --------------------------------------------------------------------------- #
def test_export_csv_header_and_row_count(tmp_path: Path) -> None:
    """First non-comment line is ``metric,value``; row count matches metrics."""
    env = make_envelope(
        model_id="meta-llama/Llama-3.1-8B",
        metrics={
            "throughput_tok_per_s": 1000.0,
            "ttft_p50_ms": 50.0,
            "ok_rate": 1.0,
        },
    )
    path = write_envelope_json(tmp_path / "env.json", env)
    result = runner.invoke(app, ["export", str(path), "--format", "csv"])
    assert result.exit_code == 0, result.output

    lines = [ln for ln in result.output.splitlines() if ln.strip()]
    non_comment = [ln for ln in lines if not ln.startswith("#")]
    assert non_comment[0] == "metric,value"

    data_rows = non_comment[1:]
    # 3 numeric metrics in the envelope → 3 data rows.
    assert len(data_rows) == 3


def test_export_csv_comment_header_contains_suite_id(tmp_path: Path) -> None:
    """CSV comment rows expose suite_id / model_id for spreadsheet context."""
    path = _write_llama_envelope(tmp_path)
    result = runner.invoke(app, ["export", str(path), "--format", "csv"])
    assert result.exit_code == 0, result.output
    assert "# suite_id=llm.inference" in result.output
    assert "# model_id=meta-llama/Llama-3.1-8B" in result.output


# --------------------------------------------------------------------------- #
# Slack                                                                       #
# --------------------------------------------------------------------------- #
def test_export_slack_format(tmp_path: Path) -> None:
    """Slack output: rocket emoji, fenced code block, percentage formatting."""
    path = _write_llama_envelope(tmp_path)
    result = runner.invoke(app, ["export", str(path), "--format", "slack"])
    assert result.exit_code == 0, result.output

    assert "\U0001f680 InferenceBench result" in result.output
    # Fenced code block — must open and close with ```.
    fence_lines = [ln for ln in result.output.splitlines() if ln.strip() == "```"]
    assert len(fence_lines) >= 2
    # ok_rate rendered as percentage, not raw 1.0.
    assert "ok_rate: 100%" in result.output
    # p50/p99 latency pairing.
    assert "ttft_p50_ms: 41.69 (p99 64.71)" in result.output
    # Power summary line.
    assert "W avg" in result.output
    assert "W peak" in result.output


# --------------------------------------------------------------------------- #
# --metric filter                                                             #
# --------------------------------------------------------------------------- #
def test_export_metric_filter_restricts_output(tmp_path: Path) -> None:
    """``--metric`` repeated: only the named metrics appear."""
    path = _write_llama_envelope(tmp_path)
    result = runner.invoke(
        app,
        [
            "export",
            str(path),
            "--format",
            "markdown",
            "--metric",
            "throughput_tok_per_s",
            "--metric",
            "ttft_p50_ms",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "throughput_tok_per_s" in result.output
    assert "ttft_p50_ms" in result.output
    # Other metrics from the envelope must be absent.
    assert "joules_per_token" not in result.output
    assert "power_avg_w" not in result.output
    assert "cost_source" not in result.output


# --------------------------------------------------------------------------- #
# --out                                                                       #
# --------------------------------------------------------------------------- #
def test_export_out_writes_to_file_and_stdout_empty(tmp_path: Path) -> None:
    """``--out`` writes to a file; stdout has no rendered output."""
    path = _write_llama_envelope(tmp_path)
    out = tmp_path / "result.md"
    result = runner.invoke(
        app,
        ["export", str(path), "--format", "markdown", "--out", str(out)],
    )
    assert result.exit_code == 0, result.output
    assert out.exists()
    contents = out.read_text()
    assert "## InferenceBench result" in contents
    # Stdout should not duplicate the rendered output.
    assert "## InferenceBench result" not in result.output


# --------------------------------------------------------------------------- #
# Error paths                                                                 #
# --------------------------------------------------------------------------- #
def test_export_invalid_format_exits_non_zero(tmp_path: Path) -> None:
    """An unknown ``--format`` value exits non-zero."""
    path = _write_llama_envelope(tmp_path)
    result = runner.invoke(app, ["export", str(path), "--format", "xml"])
    assert result.exit_code != 0


def test_export_missing_envelope_exits_2(tmp_path: Path) -> None:
    """A non-existent envelope path exits 2."""
    missing = tmp_path / "does_not_exist.json"
    result = runner.invoke(app, ["export", str(missing)])
    assert result.exit_code == 2
