"""TestBM GPU nightly integration test (ticket 0024).

This test is RUN ONLY ON THE TESTBM NIGHTLY JOB, not on PR CI. It exists to
prove the plugin's GPU-touching code paths (fingerprint collection, validate)
work on a real CUDA host. Phase 1 keeps it minimal — it does not require a
live vLLM server. If vLLM is reachable, the validate() warnings shrink; if
not, that's fine and expected on a freshly-provisioned box.

Marker: ``@pytest.mark.gpu``. Skipped automatically when CUDA isn't present
(detected via pynvml — we deliberately avoid pulling torch as a test dep).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from inferencebench.harness import collect_hardware_fingerprint
from inferencebench_llm import LLMInferencePlugin
from inferencebench_llm.schemas import (
    BenchmarkSpec,
    DatasetConfig,
    DatasetSamplingConfig,
    DriverConfig,
    EngineKind,
    RunContext,
    WarmupConfig,
)


def _smoke_spec() -> BenchmarkSpec:
    return BenchmarkSpec(
        benchmark_id="llm.inference.smoke-gpu",
        suite_version="0.0.1",
        description="GPU smoke test (nightly).",
        modality="llm",
        kind="perf",
        dataset=DatasetConfig(
            id="smoke",
            uri="builtin://fallback",
            hash="sha256:" + "0" * 64,
            sampling=DatasetSamplingConfig(n=5, seed=42),
        ),
        driver=DriverConfig(
            type="closed_loop",
            concurrency=[1],
            duration_s=1,
        ),
        slo_template="llm.relaxed",
        warmup=WarmupConfig(discard_runs=0, convergence_window=2),
    )


@pytest.mark.gpu
def test_gpu_fingerprint_and_validate(tmp_path: Path) -> None:
    """Smoke: NVML present -> fingerprint reports GPUs; plugin.validate runs cleanly."""
    try:
        import pynvml

        pynvml.nvmlInit()
    except Exception as exc:
        pytest.skip(f"CUDA / NVML not available: {exc}")

    try:
        # 1. Fingerprint should see at least one GPU when NVML init succeeded.
        hw = collect_hardware_fingerprint()
        assert len(hw.gpus) >= 1, "CUDA present but no GPUs reported by fingerprint collector"

        # 2. plugin.validate() against a (likely down) vLLM endpoint should yield
        #    at most one warning — about the engine being unreachable. That's
        #    expected on a freshly-provisioned nightly box; the point is we
        #    exercise the code path, not that vLLM is running.
        plugin = LLMInferencePlugin()
        context = RunContext(
            model_id="meta-llama/Llama-4-Maverick",
            engine_kind=EngineKind.VLLM,
            base_url="http://localhost:8000/v1",
            output_dir=tmp_path / "out",
        )
        warnings = plugin.validate(_smoke_spec(), context)
        # At most one warning — the engine-unreachable one. base_url and
        # model_id are both populated, so no other warning should fire.
        assert len(warnings) <= 1, f"unexpected warnings from validate: {warnings}"
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass
