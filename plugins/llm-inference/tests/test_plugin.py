"""Tests for the llm-inference plugin scaffold + vLLM engine adapter."""

from __future__ import annotations

from pathlib import Path

import pytest

from inferencebench_llm import BenchmarkSpec, EngineKind, LLMInferencePlugin, RunContext
from inferencebench_llm.engines import EngineUnavailableError, VLLMEngine


# --------------------------------------------------------------------------- #
# Plugin contract                                                             #
# --------------------------------------------------------------------------- #
def test_plugin_metadata() -> None:
    plugin = LLMInferencePlugin()
    assert plugin.suite_id == "llm.inference"
    assert plugin.version
    assert plugin.description


def test_plugin_lists_bundled_benchmarks() -> None:
    plugin = LLMInferencePlugin()
    specs = plugin.list_benchmarks()
    assert len(specs) >= 1
    ids = [s.benchmark_id for s in specs]
    assert "llm.inference.sharegpt-v3" in ids


def test_plugin_get_benchmark_returns_spec() -> None:
    plugin = LLMInferencePlugin()
    spec = plugin.get_benchmark("llm.inference.sharegpt-v3")
    assert isinstance(spec, BenchmarkSpec)
    assert spec.modality == "llm"
    assert spec.driver.type == "open_loop"
    assert 1 in spec.driver.rps
    assert spec.warmup.discard_runs == 3


def test_plugin_get_benchmark_missing_id_raises_keyerror() -> None:
    plugin = LLMInferencePlugin()
    with pytest.raises(KeyError):
        plugin.get_benchmark("nonexistent.benchmark")


def test_plugin_run_requires_reachable_engine() -> None:
    """``run()`` probes the engine first; unreachable endpoint raises EngineUnavailableError."""
    plugin = LLMInferencePlugin()
    spec = plugin.get_benchmark("llm.inference.sharegpt-v3")
    ctx = RunContext(
        model_id="meta-llama/Llama-4-Maverick",
        engine_kind=EngineKind.VLLM,
        base_url="http://127.0.0.1:1/v1",  # unreachable port
        output_dir=Path("/tmp/bench"),
        extra={"signing_mode": "dev", "dev_key_path": "/tmp/nope.key"},
    )
    with pytest.raises(EngineUnavailableError):
        plugin.run(spec, ctx)


# --------------------------------------------------------------------------- #
# Validation                                                                  #
# --------------------------------------------------------------------------- #
def test_validate_warns_when_vllm_base_url_missing() -> None:
    plugin = LLMInferencePlugin()
    spec = plugin.get_benchmark("llm.inference.sharegpt-v3")
    ctx = RunContext(
        model_id="m",
        engine_kind=EngineKind.VLLM,
        base_url="",
        output_dir=Path("/tmp/bench"),
    )
    warnings = plugin.validate(spec, ctx)
    assert any("base_url" in w.lower() for w in warnings)


def test_validate_warns_on_unsupported_engine() -> None:
    """TRT-LLM isn't implemented yet — validate should flag it."""
    plugin = LLMInferencePlugin()
    spec = plugin.get_benchmark("llm.inference.sharegpt-v3")
    ctx = RunContext(
        model_id="m",
        engine_kind=EngineKind.TRTLLM,
        base_url="http://localhost:8000/v1",
        output_dir=Path("/tmp/bench"),
    )
    warnings = plugin.validate(spec, ctx)
    assert any("not implemented" in w.lower() for w in warnings)


def test_run_context_rejects_empty_model_id() -> None:
    """Empty model_id is rejected at RunContext build by Pydantic — never reaches validate()."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        RunContext(
            model_id="",
            engine_kind=EngineKind.VLLM,
            base_url="http://localhost:8000/v1",
            output_dir=Path("/tmp/bench"),
        )


# --------------------------------------------------------------------------- #
# VLLMEngine                                                                  #
# --------------------------------------------------------------------------- #
def test_vllm_engine_requires_base_url() -> None:
    engine = VLLMEngine()
    ctx = RunContext(
        model_id="m",
        engine_kind=EngineKind.VLLM,
        base_url="",
        output_dir=Path("/tmp/bench"),
    )
    with pytest.raises(EngineUnavailableError, match="requires"):
        engine.probe(ctx)


def test_vllm_engine_unreachable_endpoint() -> None:
    engine = VLLMEngine()
    ctx = RunContext(
        model_id="m",
        engine_kind=EngineKind.VLLM,
        base_url="http://127.0.0.1:1/v1",
        output_dir=Path("/tmp/bench"),
    )
    with pytest.raises(EngineUnavailableError):
        engine.probe(ctx)


def test_vllm_engine_build_client_yields_openai_prefixed_model() -> None:
    engine = VLLMEngine()
    ctx = RunContext(
        model_id="meta-llama/Llama-4-Maverick",
        engine_kind=EngineKind.VLLM,
        base_url="http://localhost:8000/v1",
        output_dir=Path("/tmp/bench"),
    )
    client = engine.build_client(ctx)
    assert client.model == "openai/meta-llama/Llama-4-Maverick"
    assert client.base_url == "http://localhost:8000/v1"


def test_vllm_engine_normalises_base_url() -> None:
    """Trailing /v1 is optional; trailing slash is stripped."""
    engine = VLLMEngine()
    for url in ("http://localhost:8000", "http://localhost:8000/", "http://localhost:8000/v1"):
        ctx = RunContext(
            model_id="m",
            engine_kind=EngineKind.VLLM,
            base_url=url,
            output_dir=Path("/tmp/bench"),
        )
        c = engine.build_client(ctx)
        assert c.base_url == "http://localhost:8000/v1"
