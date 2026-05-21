"""Tests for ``bench spec`` — validate / show / lint benchmark spec YAML files."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from inferencebench.cli import app

# Wide console so Rich tables don't truncate the substrings we assert on.
runner = CliRunner(env={"COLUMNS": "240"})


# --------------------------------------------------------------------------- #
# Fixture helpers                                                             #
# --------------------------------------------------------------------------- #
_VALID_SPEC_YAML = """\
benchmark_id: llm.inference.test-bench
suite_version: 1.0.0
description: A short test benchmark for the spec command.
modality: llm
kind: perf
dataset:
  id: test-dataset
  uri: builtin://
  hash: sha256:0000000000000000000000000000000000000000000000000000000000000000
  sampling:
    n: 50
    seed: 42
driver:
  type: closed_loop
  arrival: poisson
  concurrency: [4, 16]
  duration_s: 120
slo_template: llm.standard
metrics:
  - ttft_p50_ms
  - throughput_tok_per_s
warmup:
  discard_runs: 3
  convergence_cov_threshold: 0.05
  convergence_window: 30
"""


def _write_valid_spec(tmp_path: Path) -> Path:
    path = tmp_path / "spec.yaml"
    path.write_text(_VALID_SPEC_YAML, encoding="utf-8")
    return path


def _write_short_duration_spec(tmp_path: Path) -> Path:
    """A valid spec with ``driver.duration_s = 10`` to trigger the lint warning."""
    yaml_text = _VALID_SPEC_YAML.replace("duration_s: 120", "duration_s: 10")
    path = tmp_path / "spec.yaml"
    path.write_text(yaml_text, encoding="utf-8")
    return path


def _write_no_description_spec(tmp_path: Path) -> Path:
    """A valid spec with the description line removed."""
    yaml_text = _VALID_SPEC_YAML.replace(
        "description: A short test benchmark for the spec command.\n", ""
    )
    path = tmp_path / "spec.yaml"
    path.write_text(yaml_text, encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# validate                                                                    #
# --------------------------------------------------------------------------- #
def test_spec_validate_accepts_valid_llm_inference_spec(tmp_path: Path) -> None:
    """A valid llm.inference spec parses under at least one plugin schema."""
    path = _write_valid_spec(tmp_path)
    result = runner.invoke(app, ["spec", "validate", str(path)])
    assert result.exit_code == 0, result.output
    # The plugin name is mentioned and the green check is present.
    assert "llm.inference" in result.output
    assert "valid" in result.output


def test_spec_validate_rejects_missing_benchmark_id(tmp_path: Path) -> None:
    """A YAML missing the required ``benchmark_id`` is rejected by every plugin."""
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        # No benchmark_id field at all — every BenchmarkSpec model requires it.
        "suite_version: 1.0.0\ndescription: missing-id\nmodality: llm\n",
        encoding="utf-8",
    )
    result = runner.invoke(app, ["spec", "validate", str(bad)])
    assert result.exit_code == 1, result.output
    # Every installed plugin should have rejected it.
    assert "rejected" in result.output


def test_spec_validate_missing_file_exits_2(tmp_path: Path) -> None:
    """A non-existent spec file exits 2."""
    missing = tmp_path / "does-not-exist.yaml"
    result = runner.invoke(app, ["spec", "validate", str(missing)])
    assert result.exit_code == 2


# --------------------------------------------------------------------------- #
# show                                                                        #
# --------------------------------------------------------------------------- #
def test_spec_show_prints_tree_with_core_fields(tmp_path: Path) -> None:
    """``spec show`` prints a tree mentioning each top-level spec field."""
    path = _write_valid_spec(tmp_path)
    result = runner.invoke(app, ["spec", "show", str(path)])
    assert result.exit_code == 0, result.output
    # The Rich tree mentions the field names from the parsed BenchmarkSpec.
    assert "benchmark_id" in result.output
    assert "description" in result.output
    assert "dataset" in result.output
    assert "driver" in result.output
    # The parsed value of benchmark_id shows up too.
    assert "llm.inference.test-bench" in result.output


def test_spec_show_failed_validation_exits_1(tmp_path: Path) -> None:
    """``spec show`` on an invalid spec exits 1."""
    bad = tmp_path / "bad.yaml"
    bad.write_text("not_a_real_field: 42\n", encoding="utf-8")
    result = runner.invoke(app, ["spec", "show", str(bad)])
    assert result.exit_code == 1


# --------------------------------------------------------------------------- #
# lint                                                                        #
# --------------------------------------------------------------------------- #
def test_spec_lint_flags_short_duration(tmp_path: Path) -> None:
    """``duration_s < 30`` is flagged as a soft warning; exit code stays 0."""
    path = _write_short_duration_spec(tmp_path)
    result = runner.invoke(app, ["spec", "lint", str(path)])
    assert result.exit_code == 0, result.output
    assert "duration_s" in result.output
    assert "short duration" in result.output


def test_spec_lint_flags_empty_description(tmp_path: Path) -> None:
    """An empty/missing description triggers the description warning."""
    path = _write_no_description_spec(tmp_path)
    result = runner.invoke(app, ["spec", "lint", str(path)])
    assert result.exit_code == 0, result.output
    assert "description" in result.output
    assert "empty" in result.output


def test_spec_lint_clean_spec_says_no_warnings(tmp_path: Path) -> None:
    """A clean spec produces no lint warnings and exits 0."""
    path = _write_valid_spec(tmp_path)
    result = runner.invoke(app, ["spec", "lint", str(path)])
    assert result.exit_code == 0, result.output
    assert "no warnings" in result.output
