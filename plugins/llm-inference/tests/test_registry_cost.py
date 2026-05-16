"""Tests for the cost-source wiring in :mod:`inferencebench_llm.plugin`.

When LiteLLM reports a non-zero ``cost_usd`` we record it as
``cost_source = "provider"``; otherwise we synthesize a reference cost from
the bundled pricing registry's cheapest provider and tag it
``cost_source = "registry:<provider>"``.
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
    SoftwareProvenance,
)
from inferencebench.harness.convergence import ConvergenceState
from inferencebench.harness.drivers.base import Sample
from inferencebench.harness.run import RunResult
from inferencebench_llm import LLMInferencePlugin
from inferencebench_llm.plugin import (
    _BLEND_INPUT_SHARE,
    _BLEND_OUTPUT_SHARE,
    _registry_reference_cost,
)
from inferencebench_llm.schemas import EngineKind, RunContext


# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #
def _hw_fp() -> HardwareFingerprint:
    body: dict[str, Any] = {
        "dmi_uuid": "11111111-2222-3333-4444-555555555555",
        "gpus": [
            GPU(
                model="H100-SXM5-80GB",
                pci_id="0000:01:00.0",
                serial="SN-TEST",
                vbios="96.00.74.00.01",
            )
        ],
        "cpu": CPU(model="Test CPU", microcode="0x1"),
        "memory": Memory(channels=12, speed_mts=4800, ecc=True),
        "bios": BIOS(version="1.0", resizable_bar=True, above_4g=True),
        "driver": "560.35.03",
        "cuda": "12.6",
        "nccl": "2.22.3",
    }
    placeholder = HardwareFingerprint.model_construct(
        fingerprint_sha256="0" * 64, numa={}, **body
    )
    real = placeholder.compute_fingerprint_sha256()
    return HardwareFingerprint(fingerprint_sha256=real, numa={}, **body)


def _sw_prov() -> SoftwareProvenance:
    return SoftwareProvenance(
        pip_freeze_hash="b" * 64,
        git_commit="deadbeef1234567",
    )


def _sample(*, tokens_in: int, tokens_out: int, cost_usd: float) -> Sample:
    """Build one OK sample with the requested token counts + cost."""
    return Sample(
        request_idx=0,
        arrival_ms=0.0,
        start_ms=0.0,
        ttft_ms=10.0,
        total_ms=20.0,
        tpot_ms=1.0,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cost_usd=cost_usd,
        finish_reason="stop",
        ok=True,
    )


def _run_result(samples: list[Sample]) -> RunResult:
    """Wrap samples in a RunResult with metrics computed."""
    return RunResult(
        samples=samples,
        gpu_telemetry=[],
        rapl_telemetry=[],
        convergence=ConvergenceState(
            n_seen=len(samples),
            n_warmed=len(samples),
            cov=0.01,
            converged=True,
            bailed_out=False,
        ),
        duration_s=1.0,
    ).compute_metrics(slos=[])


def _ctx(model_id: str, tmp_path: Path) -> RunContext:
    return RunContext(
        model_id=model_id,
        engine_kind=EngineKind.VLLM,
        base_url="http://localhost:8000/v1",
        output_dir=tmp_path,
    )


def _build(
    plugin: LLMInferencePlugin,
    ctx: RunContext,
    result: RunResult,
    monkeypatch: pytest.MonkeyPatch,
) -> Any:
    """Call ``_build_envelope`` with HW/SW collectors stubbed out."""
    monkeypatch.setattr(
        "inferencebench_llm.plugin.collect_hardware_fingerprint",
        lambda: _hw_fp(),
    )
    monkeypatch.setattr(
        "inferencebench_llm.plugin.collect_software_provenance",
        lambda: _sw_prov(),
    )
    spec = plugin.get_benchmark("llm.inference.sharegpt-v3")
    return plugin._build_envelope(
        spec, ctx, "0.7.2", result, dataset_hash="0" * 64
    )


# --------------------------------------------------------------------------- #
# Provider-cost path                                                          #
# --------------------------------------------------------------------------- #
def test_provider_cost_path_tagged_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """LiteLLM-reported cost → cost_source='provider', $/Mtok from sample totals."""
    plugin = LLMInferencePlugin()
    # 1000 output tokens, $0.001 → $1.00 / M output tokens.
    samples = [_sample(tokens_in=500, tokens_out=1000, cost_usd=0.001)]
    result = _run_result(samples)

    ctx = _ctx("meta-llama/Llama-3.1-8B-Instruct", tmp_path)
    env = _build(plugin, ctx, result, monkeypatch)

    assert env.metrics["cost_usd_per_million_tokens"] == pytest.approx(1.0)
    assert env.metrics["cost_source"] == "provider"


# --------------------------------------------------------------------------- #
# Registry-cost path                                                          #
# --------------------------------------------------------------------------- #
def test_registry_cost_path_picks_cheapest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No provider cost + known model → cheapest registered blended rate."""
    plugin = LLMInferencePlugin()
    samples = [_sample(tokens_in=500, tokens_out=1000, cost_usd=0.0)]
    result = _run_result(samples)

    ctx = _ctx("meta-llama/Llama-3.1-8B-Instruct", tmp_path)
    env = _build(plugin, ctx, result, monkeypatch)

    # Llama-3.1-8B-Instruct: groq (0.05 in / 0.08 out) is cheapest:
    # blended = 0.75 * 0.05 + 0.25 * 0.08 = 0.0575
    expected = _BLEND_INPUT_SHARE * 0.05 + _BLEND_OUTPUT_SHARE * 0.08
    assert env.metrics["cost_usd_per_million_tokens"] == pytest.approx(expected)
    assert env.metrics["cost_source"] == "registry:groq"


def test_registry_cost_strips_openai_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """LiteLLM-style ``openai/<hf-id>`` model id still resolves through the registry."""
    plugin = LLMInferencePlugin()
    samples = [_sample(tokens_in=500, tokens_out=1000, cost_usd=0.0)]
    result = _run_result(samples)

    ctx = _ctx("openai/meta-llama/Llama-3.1-8B-Instruct", tmp_path)
    env = _build(plugin, ctx, result, monkeypatch)

    assert "cost_usd_per_million_tokens" in env.metrics
    assert env.metrics["cost_source"] == "registry:groq"


def test_registry_cost_unknown_model_emits_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unknown model + zero provider cost → no cost metric, no cost_source."""
    plugin = LLMInferencePlugin()
    samples = [_sample(tokens_in=500, tokens_out=1000, cost_usd=0.0)]
    result = _run_result(samples)

    ctx = _ctx("mistralai/Some-Unknown-Mystery-Model-9000", tmp_path)
    env = _build(plugin, ctx, result, monkeypatch)

    assert "cost_usd_per_million_tokens" not in env.metrics
    assert "cost_source" not in env.metrics


# --------------------------------------------------------------------------- #
# Cheapest-blended-rate selection — unit-level                                #
# --------------------------------------------------------------------------- #
def test_cheapest_blended_rate_selection() -> None:
    """Given three providers, the smallest blended rate wins."""
    # Llama-3.1-70B-Instruct prices (per million tokens):
    #   together  0.88 in / 0.88 out  → blended 0.88
    #   fireworks 0.90 in / 0.90 out  → blended 0.90
    #   groq      0.59 in / 0.79 out  → blended 0.75 * 0.59 + 0.25 * 0.79 = 0.64
    result = _registry_reference_cost("meta-llama/Llama-3.1-70B-Instruct")
    assert result is not None
    rate, provider = result
    assert provider == "groq"
    assert rate == pytest.approx(0.75 * 0.59 + 0.25 * 0.79)


def test_cheapest_blended_rate_selection_synthetic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inject three providers at [0.10, 0.20, 0.30] blended → the 0.10 wins."""
    from inferencebench_llm import pricing as pricing_mod

    fake_entries = [
        pricing_mod.ModelPricing(
            provider="alpha",
            model="fake/Model-X",
            input_per_million_usd=0.30,
            output_per_million_usd=0.30,
        ),
        pricing_mod.ModelPricing(
            provider="beta",
            model="fake/Model-X",
            input_per_million_usd=0.10,
            output_per_million_usd=0.10,
        ),
        pricing_mod.ModelPricing(
            provider="gamma",
            model="fake/Model-X",
            input_per_million_usd=0.20,
            output_per_million_usd=0.20,
        ),
    ]

    def fake_providers_for(model: str) -> list[pricing_mod.ModelPricing]:
        assert model == "fake/Model-X"
        return fake_entries

    monkeypatch.setattr(
        "inferencebench_llm.plugin.providers_for", fake_providers_for
    )

    result = _registry_reference_cost("fake/Model-X")
    assert result is not None
    rate, provider = result
    assert provider == "beta"
    assert rate == pytest.approx(0.10)
