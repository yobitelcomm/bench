"""Tests for ``bench ci`` — workflow generation + validation.

Covers ``bench ci init`` (default + custom substitutions, overwrite guard,
``--force``) and ``bench ci validate`` (happy path on freshly-generated
workflows, failure modes for malformed / non-strict / non-yaml files).
"""

from __future__ import annotations

from pathlib import Path

import yaml
from typer.testing import CliRunner

from inferencebench.cli import app

runner = CliRunner(env={"COLUMNS": "240"})


# --------------------------------------------------------------------------- #
# init                                                                        #
# --------------------------------------------------------------------------- #
def test_ci_init_writes_parseable_yaml(tmp_path: Path) -> None:
    """Default ``bench ci init --out <tmp>/wf.yml`` writes a parseable file."""
    out = tmp_path / "wf.yml"
    result = runner.invoke(app, ["ci", "init", "--out", str(out)])
    assert result.exit_code == 0, result.stdout + (result.stderr or "")
    assert out.exists()
    parsed = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert isinstance(parsed, dict)
    # PyYAML 1.1 quirk: the bare key `on` becomes Python `True`.
    triggers = parsed.get("on", parsed.get(True))
    assert "pull_request" in triggers
    assert "workflow_dispatch" in triggers
    assert "bench" in parsed["jobs"]


def test_ci_init_creates_parent_dirs(tmp_path: Path) -> None:
    """A nested ``--out`` path causes parent directories to be mkdir-p'd."""
    out = tmp_path / "nested" / "deeper" / "wf.yml"
    result = runner.invoke(app, ["ci", "init", "--out", str(out)])
    assert result.exit_code == 0, result.stdout + (result.stderr or "")
    assert out.exists()


def test_ci_init_substitutes_custom_params(tmp_path: Path) -> None:
    """Non-default suite/model/engine values land in the rendered YAML body."""
    out = tmp_path / "wf.yml"
    result = runner.invoke(
        app,
        [
            "ci",
            "init",
            "--out",
            str(out),
            "--suite",
            "voice.transcribe.fleurs",
            "--model",
            "openai/whisper-large-v3",
            "--engine",
            "sglang",
            "--baseline",
            "ci/baseline.json",
            "--runner",
            "self-hosted,h100",
            "--tolerance",
            "0.1",
        ],
    )
    assert result.exit_code == 0, result.stdout + (result.stderr or "")
    body = out.read_text(encoding="utf-8")
    assert "voice.transcribe.fleurs" in body
    assert "openai/whisper-large-v3" in body
    assert "sglang" in body
    assert "ci/baseline.json" in body
    # Multi-label runner is rendered as a flow-style list.
    assert "[self-hosted, h100]" in body
    assert "--tolerance 0.1" in body


def test_ci_init_refuses_to_overwrite(tmp_path: Path) -> None:
    """Second ``init`` without ``--force`` fails with exit 1."""
    out = tmp_path / "wf.yml"
    first = runner.invoke(app, ["ci", "init", "--out", str(out)])
    assert first.exit_code == 0, first.stdout

    second = runner.invoke(app, ["ci", "init", "--out", str(out)])
    assert second.exit_code == 1
    # File contents untouched.
    assert "bench-regression" in out.read_text(encoding="utf-8")


def test_ci_init_force_overwrites(tmp_path: Path) -> None:
    """``--force`` lets the second invocation rewrite the file."""
    out = tmp_path / "wf.yml"
    first = runner.invoke(
        app,
        ["ci", "init", "--out", str(out), "--model", "first/model"],
    )
    assert first.exit_code == 0, first.stdout

    second = runner.invoke(
        app,
        ["ci", "init", "--out", str(out), "--model", "second/model", "--force"],
    )
    assert second.exit_code == 0, second.stdout
    body = out.read_text(encoding="utf-8")
    assert "second/model" in body
    assert "first/model" not in body


# --------------------------------------------------------------------------- #
# validate                                                                    #
# --------------------------------------------------------------------------- #
def test_ci_validate_passes_on_generated_workflow(tmp_path: Path) -> None:
    """A freshly-generated workflow passes every shape check."""
    out = tmp_path / "wf.yml"
    init = runner.invoke(app, ["ci", "init", "--out", str(out)])
    assert init.exit_code == 0, init.stdout

    result = runner.invoke(app, ["ci", "validate", str(out)])
    assert result.exit_code == 0, result.stdout + (result.stderr or "")
    assert "PASS" in result.stdout
    assert "All checks passed" in result.stdout


def test_ci_validate_fails_when_diff_step_missing(tmp_path: Path) -> None:
    """A workflow without a ``bench diff`` step → exit 1 with a clear list."""
    bad = tmp_path / "bad.yml"
    bad.write_text(
        """\
name: bench-regression
on:
  pull_request:
    branches: [main]
jobs:
  bench:
    runs-on: [self-hosted, gpu]
    steps:
      - uses: actions/checkout@v4
      - name: Run benchmark
        run: bench run llm.inference.sharegpt-v3 --model foo --engine vllm
""",
        encoding="utf-8",
    )
    result = runner.invoke(app, ["ci", "validate", str(bad)])
    assert result.exit_code == 1
    # The diff-check fails; the strict-check also fails (no diff command).
    assert "FAIL" in result.stdout
    assert "bench diff" in result.stdout


def test_ci_validate_fails_when_diff_missing_strict(tmp_path: Path) -> None:
    """A diff invocation without ``--strict`` is treated as a failure."""
    bad = tmp_path / "bad-no-strict.yml"
    bad.write_text(
        """\
name: bench-regression
on:
  pull_request:
    branches: [main]
jobs:
  bench:
    runs-on: [self-hosted, gpu]
    steps:
      - name: Run benchmark
        run: bench run llm.inference.sharegpt-v3 --model foo --engine vllm
      - name: Diff against baseline
        run: bench diff baseline.json new.json --tolerance 0.05
""",
        encoding="utf-8",
    )
    result = runner.invoke(app, ["ci", "validate", str(bad)])
    assert result.exit_code == 1
    # Specifically the strict check should fail.
    assert "--strict" in result.stdout


def test_ci_validate_exits_2_on_unparseable_yaml(tmp_path: Path) -> None:
    """Unparseable input → exit 2 (distinct from logical-failure exit 1)."""
    bad = tmp_path / "not-yaml.yml"
    bad.write_text("{ this is : : not [ valid }} yaml ::: \n@#$\n", encoding="utf-8")
    result = runner.invoke(app, ["ci", "validate", str(bad)])
    assert result.exit_code == 2


def test_ci_validate_exits_2_on_missing_file(tmp_path: Path) -> None:
    """Missing workflow path → exit 2."""
    result = runner.invoke(app, ["ci", "validate", str(tmp_path / "absent.yml")])
    assert result.exit_code == 2
