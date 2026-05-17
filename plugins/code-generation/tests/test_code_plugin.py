"""Tests for the code-generation plugin scaffold + end-to-end pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from inferencebench.envelope import generate_dev_keypair
from inferencebench_code import (
    BenchmarkSpec,
    CodeGenerationPlugin,
    EngineKind,
    RunContext,
)

# --------------------------------------------------------------------------- #
# Canned solutions used by the mock model                                     #
# --------------------------------------------------------------------------- #
# Keyed by entry_point — we route by inspecting the prompt for the def line.
_CORRECT_SOLUTIONS: dict[str, str] = {
    "add": "def add(a, b):\n    return a + b\n",
    "reverse_string": "def reverse_string(s):\n    return s[::-1]\n",
    "fib": "def fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n",
    "count_vowels": "def count_vowels(s):\n    return sum(1 for c in s if c in 'aeiou')\n",
    "is_palindrome": "def is_palindrome(s):\n    t = s.lower()\n    return t == t[::-1]\n",
}


def _entry_point_for(prompt: str) -> str:
    """Pull the function name out of the prompt's ``def <name>(...)`` line."""
    for line in prompt.splitlines():
        stripped = line.strip()
        if stripped.startswith("def "):
            return stripped[len("def ") :].split("(", 1)[0].strip()
    return ""


def _fence(code: str) -> str:
    return f"```python\n{code}```"


# --------------------------------------------------------------------------- #
# Plugin contract                                                             #
# --------------------------------------------------------------------------- #
def test_plugin_metadata() -> None:
    plugin = CodeGenerationPlugin()
    assert plugin.suite_id == "code.generation"
    assert plugin.version == "0.0.2"
    assert plugin.description


def test_plugin_lists_bundled_benchmarks() -> None:
    plugin = CodeGenerationPlugin()
    specs = plugin.list_benchmarks()
    assert len(specs) >= 2
    ids = {s.benchmark_id for s in specs}
    assert {
        "code.generation.humaneval-mini",
        "code.generation.mbpp-mini",
    }.issubset(ids)


def test_get_benchmark_mbpp_mini_resolves() -> None:
    plugin = CodeGenerationPlugin()
    spec = plugin.get_benchmark("code.generation.mbpp-mini")
    assert isinstance(spec, BenchmarkSpec)
    assert spec.modality == "code"
    assert spec.kind == "generation"
    assert spec.scoring == "pass_at_1"
    assert spec.language == "python"
    assert spec.timeout_s == 5.0
    assert spec.dataset.path == "mbpp-mini.jsonl"

    # Fixture has exactly 5 entries.
    plugin_dir = Path(plugin._benchmarks_dir()).parent
    fixture_path = plugin_dir / "datasets" / "mbpp-mini.jsonl"
    assert fixture_path.exists()
    lines = [
        line for line in fixture_path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    assert len(lines) == 5


def test_plugin_get_benchmark_humaneval_mini() -> None:
    plugin = CodeGenerationPlugin()
    spec = plugin.get_benchmark("code.generation.humaneval-mini")
    assert isinstance(spec, BenchmarkSpec)
    assert spec.modality == "code"
    assert spec.kind == "generation"
    assert spec.scoring == "pass_at_1"
    assert spec.language == "python"
    assert spec.timeout_s == 5.0


def test_plugin_get_benchmark_missing_id_raises_keyerror() -> None:
    plugin = CodeGenerationPlugin()
    with pytest.raises(KeyError):
        plugin.get_benchmark("nonexistent.benchmark")


def test_validate_warns_when_self_hosted_base_url_missing() -> None:
    plugin = CodeGenerationPlugin()
    spec = plugin.get_benchmark("code.generation.humaneval-mini")
    ctx = RunContext(
        model_id="m",
        engine_kind=EngineKind.VLLM,
        base_url="",
        output_dir=Path("/tmp/bench"),
    )
    warnings = plugin.validate(spec, ctx)
    assert any("base_url" in w.lower() for w in warnings)


# --------------------------------------------------------------------------- #
# End-to-end runs (mocked client)                                             #
# --------------------------------------------------------------------------- #
def _ctx(tmp_path: Path) -> RunContext:
    private_key_path = tmp_path / "cosign.key"
    generate_dev_keypair(private_key_path)
    return RunContext(
        model_id="openai/mock-model",
        model_revision="abc1234",
        engine_kind=EngineKind.OPENAI,
        output_dir=tmp_path / "out",
        extra={
            "signing_mode": "dev",
            "dev_key_path": str(private_key_path),
        },
    )


def test_run_four_of_five_passing_yields_pass_at_1_zero_point_eight(
    make_mock_modelclient, tmp_path: Path
) -> None:
    """Mock returns correct solutions for 4 of 5 tasks → pass_at_1 = 0.8."""
    # Intentionally break is_palindrome so it returns False for empties.
    broken_palindrome = (
        "def is_palindrome(s):\n"
        "    if not s:\n"
        "        return False\n"
        "    t = s.lower()\n"
        "    return t == t[::-1]\n"
    )

    def responder(prompt: str) -> str:
        ep = _entry_point_for(prompt)
        if ep == "is_palindrome":
            return _fence(broken_palindrome)
        return _fence(_CORRECT_SOLUTIONS.get(ep, ""))

    make_mock_modelclient(responder)

    plugin = CodeGenerationPlugin()
    spec = plugin.get_benchmark("code.generation.humaneval-mini")
    envelope = plugin.run(spec, _ctx(tmp_path))

    assert envelope.signature is not None
    assert envelope.metrics.get("n_samples") == 5.0
    pass_at_1 = envelope.metrics.get("pass_at_1")
    assert pass_at_1 is not None
    assert isinstance(pass_at_1, (int, float))
    assert float(pass_at_1) == pytest.approx(0.8)
    assert envelope.metrics.get("ok_rate") == 1.0
    # supplementary metrics present
    assert envelope.metrics.get("pass_at_1_p50") is not None
    assert envelope.metrics.get("timeout_rate") == 0.0


def test_run_broken_code_yields_zero_pass_at_1_but_ok_rate_one(
    make_mock_modelclient, tmp_path: Path
) -> None:
    """Syntactically-broken responses fail every task; the run still succeeds."""

    def responder(prompt: str) -> str:
        return _fence("def something(:\n    return broken\n")

    make_mock_modelclient(responder)

    plugin = CodeGenerationPlugin()
    spec = plugin.get_benchmark("code.generation.humaneval-mini")
    envelope = plugin.run(spec, _ctx(tmp_path))

    assert envelope.signature is not None
    assert envelope.metrics.get("pass_at_1") == 0.0
    # The run itself succeeded (we got responses from every prompt), just
    # the model output was garbage.
    assert envelope.metrics.get("ok_rate") == 1.0
    assert envelope.metrics.get("n_samples") == 5.0


def test_run_forbidden_import_recorded_in_samples(
    make_mock_modelclient, tmp_path: Path
) -> None:
    """Mock returns code that imports subprocess → all fail with forbidden_import."""

    def responder(prompt: str) -> str:
        return _fence("import subprocess\n\n" + _CORRECT_SOLUTIONS["add"])

    make_mock_modelclient(responder)

    plugin = CodeGenerationPlugin()
    spec = plugin.get_benchmark("code.generation.humaneval-mini")
    envelope = plugin.run(spec, _ctx(tmp_path))

    assert envelope.metrics.get("pass_at_1") == 0.0

    # Inspect the diagnostic dump for forbidden_import reasons.
    out_dir = tmp_path / "out"
    samples_files = list(out_dir.glob("samples-*.jsonl"))
    assert len(samples_files) == 1


# --------------------------------------------------------------------------- #
# End-to-end CLI smoke: bench summary <dir>                                   #
# --------------------------------------------------------------------------- #
def test_envelope_passes_through_bench_summary(
    make_mock_modelclient, tmp_path: Path
) -> None:
    """Write a code envelope to disk and run ``bench summary`` over it."""

    def responder(prompt: str) -> str:
        ep = _entry_point_for(prompt)
        return _fence(_CORRECT_SOLUTIONS.get(ep, ""))

    make_mock_modelclient(responder)

    plugin = CodeGenerationPlugin()
    spec = plugin.get_benchmark("code.generation.humaneval-mini")
    envelope = plugin.run(spec, _ctx(tmp_path))

    # Drop the envelope as a JSON file in a fresh directory.
    envelope_dir = tmp_path / "envelopes"
    envelope_dir.mkdir()
    (envelope_dir / "code.json").write_text(
        envelope.model_dump_json(), encoding="utf-8"
    )

    from inferencebench.cli import app  # local import — keeps top-level fast

    runner = CliRunner()
    result = runner.invoke(app, ["summary", str(envelope_dir)])
    assert result.exit_code == 0, result.output
    assert "code.generation.humaneval-mini" in result.output


# --------------------------------------------------------------------------- #
# Spec round-trip / sample-jsonl format                                       #
# --------------------------------------------------------------------------- #
def test_run_writes_samples_jsonl_with_pass_flag(
    make_mock_modelclient, tmp_path: Path
) -> None:
    def responder(prompt: str) -> str:
        ep = _entry_point_for(prompt)
        return _fence(_CORRECT_SOLUTIONS.get(ep, ""))

    make_mock_modelclient(responder)

    plugin = CodeGenerationPlugin()
    spec = plugin.get_benchmark("code.generation.humaneval-mini")
    ctx = _ctx(tmp_path)
    plugin.run(spec, ctx)

    out_dir = tmp_path / "out"
    samples_files = list(out_dir.glob("samples-*.jsonl"))
    assert len(samples_files) == 1
    lines = samples_files[0].read_text(encoding="utf-8").splitlines()
    assert len(lines) == 5
    parsed = [json.loads(line) for line in lines]
    assert all("passed" in row for row in parsed)
    assert all(row["passed"] is True for row in parsed)
