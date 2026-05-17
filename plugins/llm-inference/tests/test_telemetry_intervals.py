"""Tests for ``RunContext.extra`` telemetry overrides reaching ``BenchmarkRun``.

The plugin's ``run()`` is supposed to honour two optional keys in
``context.extra``:

- ``nvml_interval_ms`` — defaults to 50
- ``rapl_interval_ms`` — defaults to 100

These tests monkeypatch ``BenchmarkRun.execute`` to capture the constructed
instance (so we don't actually need a live engine), and assert the intervals
flow through.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

from inferencebench.harness.convergence import ConvergenceState
from inferencebench.harness.run import BenchmarkRun, RunResult
from inferencebench_llm import EngineKind, LLMInferencePlugin, RunContext

if TYPE_CHECKING:
    import pytest


def _empty_run_result() -> RunResult:
    """Build a RunResult with zero samples — enough to let ``run()`` finish."""
    return RunResult(
        samples=[],
        gpu_telemetry=[],
        rapl_telemetry=[],
        convergence=ConvergenceState(
            n_seen=0,
            n_warmed=0,
            cov=float("nan"),
            converged=False,
            bailed_out=False,
        ),
        duration_s=1.0,
    )


def _patch_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace VLLMEngine.probe + build_client so ``run()`` can proceed offline."""
    from inferencebench_llm.engines import VLLMEngine

    class _FakeClient:
        def complete(
            self, prompt: str, *, max_tokens: int = 128, stream: bool = True
        ) -> Any:
            raise RuntimeError("client never invoked: driver short-circuited in test")

    monkeypatch.setattr(VLLMEngine, "probe", lambda self, ctx: "fake-0.0.0")
    monkeypatch.setattr(VLLMEngine, "build_client", lambda self, ctx: _FakeClient())


def _patch_signing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Short-circuit envelope signing so the test doesn't touch disk for a key."""
    import inferencebench_llm.plugin as plugin_mod

    monkeypatch.setattr(plugin_mod, "sign_envelope", lambda env, **_kw: env)


def _capture_bench_run(
    captured: dict[str, Any],
) -> Any:
    """Return a BenchmarkRun.execute replacement that snapshots ``self``."""

    def _execute(self: BenchmarkRun) -> RunResult:
        captured["nvml_interval_ms"] = self.nvml_interval_ms
        captured["rapl_interval_ms"] = self.rapl_interval_ms
        return _empty_run_result()

    return _execute


def _run_plugin(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    extra: dict[str, str | int | float | bool],
) -> dict[str, Any]:
    """Drive ``LLMInferencePlugin.run`` with the given ``extra`` and capture intervals."""
    _patch_engine(monkeypatch)
    _patch_signing(monkeypatch)

    plugin = LLMInferencePlugin()
    spec = plugin.get_benchmark("llm.inference.sharegpt-v3")

    base_extra: dict[str, str | int | float | bool] = {
        "signing_mode": "dev",
        "dev_key_path": str(tmp_path / "fake.key"),
    }
    base_extra.update(extra)

    ctx = RunContext(
        model_id="meta-llama/Llama-3.1-8B-Instruct",
        engine_kind=EngineKind.VLLM,
        base_url="http://localhost:8000/v1",
        output_dir=tmp_path / "out",
        extra=base_extra,
    )

    captured: dict[str, Any] = {}
    with patch.object(BenchmarkRun, "execute", _capture_bench_run(captured)):
        plugin.run(spec, ctx)
    return captured


def test_telemetry_intervals_propagate_from_extra(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``nvml_interval_ms`` and ``rapl_interval_ms`` in extra reach BenchmarkRun."""
    captured = _run_plugin(
        monkeypatch,
        tmp_path,
        extra={"nvml_interval_ms": 10, "rapl_interval_ms": 25},
    )
    assert captured["nvml_interval_ms"] == 10
    assert captured["rapl_interval_ms"] == 25


def test_telemetry_intervals_default_when_absent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When extra omits the keys, BenchmarkRun receives the documented defaults."""
    captured = _run_plugin(monkeypatch, tmp_path, extra={})
    assert captured["nvml_interval_ms"] == 50
    assert captured["rapl_interval_ms"] == 100
