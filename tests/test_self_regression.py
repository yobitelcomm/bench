"""Tests for the bench-on-bench self-regression workflow.

Covers the trio that makes this product credible:

1. ``scripts/self_regression_bench.py`` runs end-to-end on CPU and produces
   exactly one signed envelope.
2. ``.github/workflows/self-regression.yml`` parses as YAML and wires up the
   key steps (synthetic bench runner + ``bench diff --strict``).
3. ``.bench/baseline.json`` is a committed, valid envelope.
4. ``.gitignore`` does NOT blanket-ignore ``.bench/`` — only the private
   ``.bench/*.key`` is excluded, so ``baseline.json`` and ``cosign.pub`` can
   be committed.
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import yaml

from inferencebench.envelope import Envelope

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "self_regression_bench.py"
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "self-regression.yml"
BASELINE = REPO_ROOT / ".bench" / "baseline.json"
GITIGNORE = REPO_ROOT / ".gitignore"


# --------------------------------------------------------------------------- #
# Script                                                                      #
# --------------------------------------------------------------------------- #
def test_script_exists_and_is_executable() -> None:
    """The runner script is present and has the user-executable bit set."""
    assert SCRIPT.exists(), f"missing: {SCRIPT}"
    mode = SCRIPT.stat().st_mode
    assert mode & stat.S_IXUSR, "scripts/self_regression_bench.py is not executable"


def test_script_produces_one_signed_envelope(tmp_path: Path) -> None:
    """Invoking the script in a subprocess writes exactly one signed envelope."""
    output = tmp_path / "out"
    dev_key = tmp_path / "cosign.key"
    env = os.environ.copy()
    # Honor the workspace VIRTUAL_ENV if the parent test process is using uv;
    # nothing extra required — the script imports inferencebench from sys.path.
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--output",
            str(output),
            "--dev-key",
            str(dev_key),
            "--duration-s",
            "1",
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert result.returncode == 0, (
        f"script exited {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    envelopes = sorted(output.glob("*.json"))
    assert len(envelopes) == 1, f"expected 1 envelope, found {len(envelopes)}: {envelopes}"

    raw = json.loads(envelopes[0].read_text(encoding="utf-8"))
    envelope = Envelope.model_validate(raw)
    assert envelope.signature is not None, "envelope must be signed"
    assert envelope.signature.bundle, "signature bundle must be populated"


def test_script_envelope_has_canonical_suite_id(tmp_path: Path) -> None:
    """The produced envelope uses the sharegpt-v3 suite_id so it's diff-comparable."""
    output = tmp_path / "out"
    dev_key = tmp_path / "cosign.key"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--output",
            str(output),
            "--dev-key",
            str(dev_key),
            "--duration-s",
            "1",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    (envelope_path,) = sorted(output.glob("*.json"))
    envelope = Envelope.model_validate_json(envelope_path.read_text(encoding="utf-8"))
    assert envelope.suite_id == "llm.inference.sharegpt-v3"


# --------------------------------------------------------------------------- #
# Workflow                                                                    #
# --------------------------------------------------------------------------- #
def test_workflow_file_exists_and_parses_as_yaml() -> None:
    """``.github/workflows/self-regression.yml`` is valid YAML with a jobs block."""
    assert WORKFLOW.exists(), f"missing: {WORKFLOW}"
    parsed = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    assert isinstance(parsed, dict), "workflow YAML must be a mapping"
    # PyYAML 1.1 quirk: bare key `on` is parsed as Python True.
    triggers = parsed.get("on", parsed.get(True))
    assert triggers is not None, "workflow must declare triggers"
    assert "pull_request" in triggers
    assert "jobs" in parsed
    assert "bench" in parsed["jobs"]


def test_workflow_runs_bench_and_diff_strict() -> None:
    """The workflow body invokes the synthetic runner AND ``bench diff --strict``."""
    body = WORKFLOW.read_text(encoding="utf-8")
    # Synthetic bench runner (the bench-run equivalent for CPU-only CI).
    assert "scripts/self_regression_bench.py" in body, "workflow must invoke the synthetic runner"
    # Strict diff gate.
    assert "bench diff" in body, "workflow must call `bench diff`"
    assert "--strict" in body, "workflow's diff step must use --strict"
    assert "--tolerance" in body, "workflow must declare a tolerance band"


# --------------------------------------------------------------------------- #
# Baseline                                                                    #
# --------------------------------------------------------------------------- #
def test_baseline_envelope_exists_and_is_valid() -> None:
    """``.bench/baseline.json`` exists and parses cleanly as an Envelope."""
    assert BASELINE.exists(), (
        f"missing baseline at {BASELINE} — "
        "run `uv run python scripts/self_regression_bench.py "
        "--output .bench/results --dev-key .bench/cosign.key` "
        "and copy the result to .bench/baseline.json"
    )
    envelope = Envelope.model_validate_json(BASELINE.read_text(encoding="utf-8"))
    assert envelope.signature is not None
    assert envelope.metrics


# --------------------------------------------------------------------------- #
# .gitignore                                                                  #
# --------------------------------------------------------------------------- #
def test_gitignore_does_not_blanket_ignore_dot_bench() -> None:
    """``.gitignore`` must NOT have a bare ``.bench/`` rule.

    A blanket rule would block ``.bench/baseline.json`` from being committed,
    defeating the regression-gating workflow.
    """
    lines = [
        ln.strip()
        for ln in GITIGNORE.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]
    assert ".bench/" not in lines, (
        "`.gitignore` blanket-ignores `.bench/` — narrow it to `.bench/*.key`"
    )
    assert ".bench" not in lines, (
        "`.gitignore` blanket-ignores `.bench` — narrow it to `.bench/*.key`"
    )
    # The private-key carve-out must still be present.
    assert ".bench/*.key" in lines, (
        "expected `.bench/*.key` rule to keep private signing keys out of the repo"
    )
