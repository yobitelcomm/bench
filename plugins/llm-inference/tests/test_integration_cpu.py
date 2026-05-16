"""TinyLlama-style CPU integration smoke test (ticket 0023, PR CI gate).

This is the end-to-end smoke test that gates every PR. It exercises the full
plugin path — spec parsing, driver, harness orchestration, envelope build,
dev-key signing, and verification — WITHOUT a real GPU or a real LLM.

The LiteLLM call is mocked at two layers:
1. ``VLLMEngine.probe`` returns a fake version string instead of hitting HTTP.
2. ``VLLMEngine.build_client`` returns a stub client whose ``.complete()``
   yields a deterministic :class:`CompletionResult` (~50ms latency, 10 tokens).

The test asserts the returned :class:`Envelope`:
- has a non-empty signature block
- contains at least one non-zero metric
- has a 64-char hex ``content_hash()``
- verifies successfully against the freshly-generated dev public key

Marker: ``@pytest.mark.integration``. Runs in <5 minutes on CPU.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

from inferencebench.envelope import generate_dev_keypair, verify_envelope
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


# --------------------------------------------------------------------------- #
# Stub client                                                                 #
# --------------------------------------------------------------------------- #
class _StubClient:
    """Stand-in for :class:`ModelClient`. Returns a fixed CompletionResult.

    Deterministic so the test is hermetic — no wall-clock or RNG dependence
    other than the harness's own scheduling.
    """

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


# --------------------------------------------------------------------------- #
# Spec builder                                                                #
# --------------------------------------------------------------------------- #
def _smoke_spec() -> BenchmarkSpec:
    """Tiny in-memory spec — closed-loop, 2 concurrency, 2 seconds, builtin prompts."""
    return BenchmarkSpec(
        benchmark_id="llm.inference.smoke",
        suite_version="0.0.1",
        description="CPU smoke test (mocked engine).",
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
            concurrency=[2],
            duration_s=2,
        ),
        slo_template="llm.relaxed",
        warmup=WarmupConfig(discard_runs=0, convergence_window=2),
    )


# --------------------------------------------------------------------------- #
# The integration test                                                        #
# --------------------------------------------------------------------------- #
@pytest.mark.integration
def test_plugin_run_end_to_end_with_mocked_engine(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """End-to-end: build spec, run plugin, sign envelope, verify signature."""
    # 1. Patch the vLLM engine so no network is touched.
    monkeypatch.setattr(VLLMEngine, "probe", lambda self, ctx: "vllm-mock-1.0")
    monkeypatch.setattr(VLLMEngine, "build_client", lambda self, ctx: _StubClient())

    # 2. Generate an ed25519 dev keypair for signing.
    private_key_path = tmp_path / "cosign.key"
    public_key_path = tmp_path / "cosign.pub"
    generate_dev_keypair(private_key_path)
    assert private_key_path.exists()
    assert public_key_path.exists()

    # 3. Build the RunContext with dev-key signing wired in.
    context = RunContext(
        model_id="stub/tinyllama-cpu",
        model_revision="abc1234",
        engine_kind=EngineKind.VLLM,
        base_url="http://stub:8000/v1",
        output_dir=tmp_path / "out",
        extra={
            "signing_mode": "dev",
            "dev_key_path": str(private_key_path),
        },
    )

    # 4. Execute. This drives the full path: spec -> driver -> harness ->
    #    envelope -> dev-key signing.
    plugin = LLMInferencePlugin()
    envelope = plugin.run(_smoke_spec(), context)

    # 5. Signature is present and non-empty.
    assert envelope.signature is not None
    assert envelope.signature.method == "dev-key"
    assert envelope.signature.bundle  # base64 signature blob

    # 6. content_hash() is a 64-char hex string.
    content_hash = envelope.content_hash()
    assert re.fullmatch(r"[0-9a-f]{64}", content_hash) is not None

    # 7. At least one metric is populated and non-zero. The mocked client returns
    #    10 tokens_out per sample, so ``n_samples`` (or another metric) is > 0.
    assert envelope.metrics, "expected at least one metric in the envelope"
    assert any(
        (v is not None and float(v) > 0) for v in envelope.metrics.values()
    ), f"expected at least one positive metric; got {envelope.metrics}"

    # 8. Verification against the freshly-generated public key succeeds.
    result = verify_envelope(envelope, dev_public_key_path=public_key_path)
    assert result.ok, f"verification failed: {result.reason}"
    assert result.method == "dev-key"
