"""End-to-end test: ``LLMInferencePlugin.run`` emits ``slo_*`` envelope keys.

We monkeypatch the plugin's hardware-fingerprint collector to return an
RTX-4090 fingerprint regardless of what the host actually is, then run the
plugin with a mocked engine (same approach as ``test_integration_cpu``) and
assert the resulting envelope's metrics contain the rescaled SLO thresholds.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from inferencebench.envelope import (
    BIOS,
    CPU,
    GPU,
    HardwareFingerprint,
    Memory,
    generate_dev_keypair,
)
from inferencebench.harness import CompletionResult
from inferencebench_llm import LLMInferencePlugin
from inferencebench_llm.engines.vllm import VLLMEngine
from inferencebench_llm.schemas import (
    BenchmarkSpec,
    DatasetConfig,
    DatasetSamplingConfig,
    DriverConfig,
    EngineKind,
    RunContext,
    WarmupConfig,
)


class _StubClient:
    model = "openai/tinyllama-stub"
    base_url = "http://stub/v1"

    def complete(self, prompt: str, **kwargs: Any) -> CompletionResult:
        return CompletionResult(
            text="ok " * 10,
            tokens_in=8,
            tokens_out=10,
            ttft_ms=20.0,
            total_ms=50.0,
            tpot_ms=3.0,
            cost_usd=0.0,
            finish_reason="stop",
            token_source="provider",
        )


def _rtx_4090_fingerprint() -> HardwareFingerprint:
    body: dict[str, Any] = {
        "dmi_uuid": "11111111-2222-3333-4444-555555555555",
        "gpus": [
            GPU(
                model="NVIDIA GeForce RTX 4090",
                pci_id="0000:01:00.0",
                serial="0000000001",
                vbios="95.02.3c.00.91",
            )
        ],
        "cpu": CPU(model="AMD Ryzen 9 7950X", microcode="0x0a601203"),
        "memory": Memory(channels=2, speed_mts=6000, ecc=False),
        "bios": BIOS(version="F8", resizable_bar=True, above_4g=True),
        "driver": "550.54.15",
        "cuda": "12.4",
        "nccl": "",
    }
    placeholder = HardwareFingerprint.model_construct(
        fingerprint_sha256="0" * 64, numa={}, **body
    )
    real = placeholder.compute_fingerprint_sha256()
    return HardwareFingerprint(fingerprint_sha256=real, numa={}, **body)


def _spec() -> BenchmarkSpec:
    return BenchmarkSpec(
        benchmark_id="llm.inference.slo-resolution",
        suite_version="0.0.1",
        description="SLO resolution test (mocked engine).",
        modality="llm",
        kind="perf",
        dataset=DatasetConfig(
            id="smoke",
            uri="builtin://fallback",
            hash="sha256:" + "0" * 64,
            sampling=DatasetSamplingConfig(n=5, seed=42),
        ),
        driver=DriverConfig(type="closed_loop", concurrency=[2], duration_s=2),
        slo_template="llm.standard",
        warmup=WarmupConfig(discard_runs=0, convergence_window=2),
    )


@pytest.mark.integration
def test_run_emits_resolved_slo_metrics_for_rtx_4090(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An RTX 4090 fingerprint produces 1.8x-scaled SLO thresholds in metrics."""
    monkeypatch.setattr(VLLMEngine, "probe", lambda self, ctx: "vllm-mock-1.0")
    monkeypatch.setattr(VLLMEngine, "build_client", lambda self, ctx: _StubClient())

    # Both the plugin and (via the same import) the harness fingerprint helper
    # need to return our synthetic RTX-4090 host. Patch the binding the plugin
    # module pulled in at import time.
    monkeypatch.setattr(
        "inferencebench_llm.plugin.collect_hardware_fingerprint",
        lambda: _rtx_4090_fingerprint(),
    )

    dev_key = tmp_path / "cosign.key"
    generate_dev_keypair(dev_key)

    context = RunContext(
        model_id="stub/tinyllama-rtx4090",
        model_revision="abc1234",
        engine_kind=EngineKind.VLLM,
        base_url="http://stub:8000/v1",
        output_dir=tmp_path / "out",
        extra={"signing_mode": "dev", "dev_key_path": str(dev_key)},
    )

    envelope = LLMInferencePlugin().run(_spec(), context)

    assert envelope.metrics.get("slo_hardware_class") == "rtx-4090"
    resolved = envelope.metrics.get("slo_template_resolved")
    assert isinstance(resolved, str)
    # 1.8x of the llm.standard base => 360 / 90 / 5400
    assert "ttft<360ms" in resolved
    assert "tpot<90ms" in resolved
    assert "total<5400ms" in resolved
