"""Tests for ``bench run --all-benchmarks``.

These exercise the run-every-spec-the-plugin-exposes dispatch path. Real
benchmark execution is stubbed by monkeypatching
``LLMInferencePlugin.run`` so the tests never hit a live engine.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from _helpers import make_envelope  # type: ignore[import-not-found]
from typer.testing import CliRunner

from inferencebench.cli import app
from inferencebench.envelope import SigningMode, generate_dev_keypair, sign_envelope

if TYPE_CHECKING:
    import pytest

runner = CliRunner(env={"COLUMNS": "240"})


def _dev_key(tmp_path: Path) -> Path:
    priv, _pub = generate_dev_keypair(tmp_path / "cosign.key")
    return priv


def _install_fake_run(
    monkeypatch: pytest.MonkeyPatch,
    *,
    ok_rate: float = 1.0,
    fail_benchmark_ids: tuple[str, ...] = (),
) -> list[dict[str, Any]]:
    """Stub LLMInferencePlugin.run with a fake signed-envelope producer.

    Returns a list the test can inspect to see which specs were invoked.
    """
    calls: list[dict[str, Any]] = []

    from inferencebench_llm.plugin import LLMInferencePlugin

    def fake_run(self: Any, spec: Any, context: Any) -> Any:  # noqa: ARG001
        calls.append({"benchmark_id": spec.benchmark_id, "extra": dict(context.extra)})

        if spec.benchmark_id in fail_benchmark_ids:
            msg = f"forced failure for {spec.benchmark_id}"
            raise RuntimeError(msg)

        # Distinct run_id per benchmark so content_hashes differ between specs.
        salt = abs(hash(spec.benchmark_id)) % 100000
        env = make_envelope(
            model_id=f"fake-model-{spec.benchmark_id}",
            run_id=f"01934567-89ab-7000-8000-0000000{salt:05d}",
            metrics={
                "throughput_tok_per_s": 1234.0,
                "ttft_p50_ms": 100.0,
                "ttft_p99_ms": 250.0,
                "tpot_p50_ms": 20.0,
                "total_p50_ms": 1500.0,
                "ok_rate": ok_rate,
                "compliance_rate": 0.97,
            },
        )
        dev_key_path = context.extra.get("dev_key_path")
        if dev_key_path:
            return sign_envelope(env, mode=SigningMode.DEV, dev_key_path=Path(str(dev_key_path)))
        return env

    def fake_validate(self: Any, spec: Any, context: Any) -> list[str]:  # noqa: ARG001
        return []

    monkeypatch.setattr(LLMInferencePlugin, "run", fake_run)
    monkeypatch.setattr(LLMInferencePlugin, "validate", fake_validate)
    return calls


def _expected_benchmark_ids() -> list[str]:
    from inferencebench_llm.plugin import LLMInferencePlugin

    return [s.benchmark_id for s in LLMInferencePlugin().list_benchmarks()]


# --------------------------------------------------------------------------- #
# Tests                                                                       #
# --------------------------------------------------------------------------- #
def test_all_benchmarks_invokes_run_once_per_spec(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`--all-benchmarks` calls plugin.run() once per BenchmarkSpec."""
    dev_key = _dev_key(tmp_path)
    calls = _install_fake_run(monkeypatch)
    out_dir = tmp_path / "results"

    expected_ids = _expected_benchmark_ids()
    assert len(expected_ids) >= 1, "expected the llm.inference plugin to expose specs"

    result = runner.invoke(
        app,
        [
            "run",
            "llm.inference",
            "--model",
            "fake/model",
            "--base-url",
            "http://localhost:8000/v1",
            "--all-benchmarks",
            "--signing-mode",
            "dev",
            "--dev-key",
            str(dev_key),
            "--output",
            str(out_dir),
        ],
    )
    combined = result.stdout + (result.stderr or "")
    assert result.exit_code == 0, combined
    assert len(calls) == len(expected_ids)
    assert {c["benchmark_id"] for c in calls} == set(expected_ids)

    # One envelope per spec, prefixed by the dotless slug.
    files = sorted(p.name for p in out_dir.glob("*.json"))
    assert len(files) == len(expected_ids), files
    for bench_id in expected_ids:
        slug = bench_id.replace(".", "-")
        assert any(f.startswith(slug + "-") for f in files), (slug, files)


def test_all_benchmarks_mutex_with_sweep(tmp_path: Path) -> None:
    """`--all-benchmarks --sweep 1` exits non-zero with a clear error."""
    dev_key = _dev_key(tmp_path)
    result = runner.invoke(
        app,
        [
            "run",
            "llm.inference",
            "--model",
            "fake/model",
            "--all-benchmarks",
            "--sweep",
            "1",
            "--signing-mode",
            "dev",
            "--dev-key",
            str(dev_key),
        ],
    )
    assert result.exit_code != 0
    combined = result.stdout + (result.stderr or "")
    assert "mutually exclusive" in combined.lower()
    assert "--all-benchmarks" in combined


def test_all_benchmarks_mutex_with_rps_sweep(tmp_path: Path) -> None:
    """`--all-benchmarks --rps-sweep 1` exits non-zero."""
    dev_key = _dev_key(tmp_path)
    result = runner.invoke(
        app,
        [
            "run",
            "llm.inference",
            "--model",
            "fake/model",
            "--all-benchmarks",
            "--rps-sweep",
            "1",
            "--signing-mode",
            "dev",
            "--dev-key",
            str(dev_key),
        ],
    )
    assert result.exit_code != 0
    combined = result.stdout + (result.stderr or "")
    assert "mutually exclusive" in combined.lower()


def test_all_benchmarks_with_fully_qualified_suite_id_errors(
    tmp_path: Path,
) -> None:
    """`--all-benchmarks` rejects a fully-qualified benchmark id."""
    dev_key = _dev_key(tmp_path)
    result = runner.invoke(
        app,
        [
            "run",
            "llm.inference.sharegpt-v3",
            "--model",
            "fake/model",
            "--all-benchmarks",
            "--signing-mode",
            "dev",
            "--dev-key",
            str(dev_key),
        ],
    )
    assert result.exit_code != 0
    combined = result.stdout + (result.stderr or "")
    assert "llm.inference.sharegpt-v3" in combined


def test_all_benchmarks_mutex_with_list(tmp_path: Path) -> None:
    """`--all-benchmarks --list` is rejected."""
    dev_key = _dev_key(tmp_path)
    result = runner.invoke(
        app,
        [
            "run",
            "llm.inference",
            "--model",
            "fake/model",
            "--all-benchmarks",
            "--list",
            "--signing-mode",
            "dev",
            "--dev-key",
            str(dev_key),
        ],
    )
    assert result.exit_code != 0
    combined = result.stdout + (result.stderr or "")
    assert "mutually exclusive" in combined.lower()


def test_all_benchmarks_continues_past_per_benchmark_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A failing spec doesn't halt the others; exit 0 if at least one passes."""
    dev_key = _dev_key(tmp_path)
    expected_ids = _expected_benchmark_ids()
    assert len(expected_ids) >= 2, "need at least 2 specs for this test"

    fail_id = expected_ids[0]
    calls = _install_fake_run(monkeypatch, fail_benchmark_ids=(fail_id,))
    out_dir = tmp_path / "results"

    result = runner.invoke(
        app,
        [
            "run",
            "llm.inference",
            "--model",
            "fake/model",
            "--base-url",
            "http://localhost:8000/v1",
            "--all-benchmarks",
            "--signing-mode",
            "dev",
            "--dev-key",
            str(dev_key),
            "--output",
            str(out_dir),
        ],
    )
    combined = result.stdout + (result.stderr or "")
    assert result.exit_code == 0, combined
    # Every spec was attempted.
    assert {c["benchmark_id"] for c in calls} == set(expected_ids)
    # Envelopes written for the non-failing specs only.
    files = sorted(p.name for p in out_dir.glob("*.json"))
    assert len(files) == len(expected_ids) - 1, files
    fail_slug = fail_id.replace(".", "-")
    assert not any(f.startswith(fail_slug + "-") for f in files), files


def teardown_module() -> None:
    if os.path.exists("cosign.key"):
        os.unlink("cosign.key")
