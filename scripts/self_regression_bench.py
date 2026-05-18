#!/usr/bin/env python3
r"""Synthetic bench-on-bench runner used by ``.github/workflows/self-regression.yml``.

Drives the ``llm-inference`` plugin end-to-end with a mocked LiteLLM client so
the entire harness path (driver -> sample collection -> envelope build -> dev
signing) runs on CPU-only GitHub-hosted runners. Produces ONE signed envelope
per invocation, written to ``<output>/<content_hash[:12]>.json``.

Pattern mirrors ``plugins/llm-inference/tests/test_integration_cpu.py``: we
monkey-patch :meth:`VLLMEngine.probe` and :meth:`VLLMEngine.build_client` so
no network is touched, then invoke ``LLMInferencePlugin.run`` against a small
in-memory ``BenchmarkSpec``.

Usage::

    uv run python scripts/self_regression_bench.py \
        --output .bench/results \
        --dev-key .bench/cosign.key
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from inferencebench.envelope import generate_dev_keypair
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
    """Stand-in for :class:`ModelClient`.

    Returns a deterministic :class:`CompletionResult` so the run is hermetic.
    Matches the shape of the production ``VLLMEngine.build_client`` return so
    the plugin's harness wiring is exercised end-to-end without a real LLM.
    """

    model = "openai/tinyllama-stub"
    base_url = "http://stub/v1"

    def complete(self, prompt: str, **kwargs: Any) -> CompletionResult:  # noqa: ARG002, ANN401
        return CompletionResult(
            text="ok " * 10,
            tokens_in=8,
            tokens_out=10,
            ttft_ms=20.0,
            total_ms=50.0,
            tpot_ms=3.0,
            cost_usd=0.0,
            finish_reason="stop",
            token_source="provider",  # noqa: S106 — token_source is a free-text tag, not a secret
        )


# --------------------------------------------------------------------------- #
# Spec builder                                                                #
# --------------------------------------------------------------------------- #
def _synthetic_spec(duration_s: int) -> BenchmarkSpec:
    """Build an in-memory ``llm.inference.sharegpt-v3`` lookalike spec.

    Closed-loop, concurrency=2, builtin prompts. The ``benchmark_id`` matches
    the canonical sharegpt-v3 suite so the produced envelope is comparable to
    other sharegpt-v3 baselines via ``bench diff``.
    """
    return BenchmarkSpec(
        benchmark_id="llm.inference.sharegpt-v3",
        suite_version="0.0.1",
        description="Synthetic CPU regression run (mocked engine).",
        modality="llm",
        kind="perf",
        dataset=DatasetConfig(
            id="self-regression-synthetic",
            uri="builtin://fallback",
            hash="sha256:" + "0" * 64,
            sampling=DatasetSamplingConfig(n=5, seed=42),
        ),
        driver=DriverConfig(
            type="closed_loop",
            concurrency=[2],
            duration_s=duration_s,
        ),
        slo_template="llm.relaxed",
        warmup=WarmupConfig(discard_runs=0, convergence_window=2),
    )


# --------------------------------------------------------------------------- #
# Engine patching                                                             #
# --------------------------------------------------------------------------- #
def _mock_probe(self: VLLMEngine, ctx: RunContext) -> str:  # noqa: ARG001
    """Replacement for :meth:`VLLMEngine.probe` — no network."""
    return "vllm-mock-1.0"


def _mock_build_client(self: VLLMEngine, ctx: RunContext) -> _StubClient:  # noqa: ARG001
    """Replacement for :meth:`VLLMEngine.build_client` — returns a stub."""
    return _StubClient()


def _patch_vllm_engine() -> None:
    """Monkey-patch :class:`VLLMEngine` so no network is touched."""
    VLLMEngine.probe = _mock_probe  # type: ignore[method-assign,assignment]
    VLLMEngine.build_client = _mock_build_client  # type: ignore[method-assign,assignment]


def _ensure_dev_key(dev_key_path: Path) -> None:
    """Generate the ed25519 keypair if it doesn't already exist."""
    if dev_key_path.exists():
        return
    dev_key_path.parent.mkdir(parents=True, exist_ok=True)
    generate_dev_keypair(dev_key_path)


def run(output_dir: Path, dev_key_path: Path, duration_s: int) -> Path:
    """Execute the synthetic bench and return the path to the written envelope."""
    output_dir.mkdir(parents=True, exist_ok=True)
    _ensure_dev_key(dev_key_path)
    _patch_vllm_engine()

    spec = _synthetic_spec(duration_s)
    context = RunContext(
        model_id="stub/tinyllama-cpu",
        model_revision="abc1234",
        engine_kind=EngineKind.VLLM,
        base_url="http://stub:8000/v1",
        output_dir=output_dir,
        extra={
            "signing_mode": "dev",
            "dev_key_path": str(dev_key_path),
        },
    )

    plugin = LLMInferencePlugin()
    envelope = plugin.run(spec, context)

    content_hash = envelope.content_hash()
    out_path = output_dir / f"{content_hash[:12]}.json"
    out_path.write_text(
        envelope.model_dump_json(indent=2, exclude_none=False),
        encoding="utf-8",
    )
    return out_path


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Directory to write the signed envelope into.",
    )
    parser.add_argument(
        "--dev-key",
        required=True,
        type=Path,
        help="Path to the ed25519 dev signing key (generated if absent).",
    )
    parser.add_argument(
        "--duration-s",
        type=int,
        default=5,
        help="Driver duration in seconds (default: 5).",
    )
    args = parser.parse_args(argv)

    out_path = run(
        output_dir=args.output,
        dev_key_path=args.dev_key,
        duration_s=args.duration_s,
    )
    print(f"wrote envelope: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
