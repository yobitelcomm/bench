"""Tests for the bundled benchmark spec YAMLs."""

from __future__ import annotations

import pytest

from inferencebench_llm import LLMInferencePlugin
from inferencebench_llm.schemas import BenchmarkSpec


def test_plugin_lists_at_least_three_specs() -> None:
    plugin = LLMInferencePlugin()
    specs = plugin.list_benchmarks()
    ids = [s.benchmark_id for s in specs]
    assert "llm.inference.sharegpt-v3" in ids
    assert "llm.inference.chatbot-short" in ids
    assert "llm.inference.long-context" in ids
    assert len(specs) >= 3


def test_each_spec_validates_against_benchmark_spec() -> None:
    plugin = LLMInferencePlugin()
    for spec in plugin.list_benchmarks():
        roundtripped = BenchmarkSpec.model_validate(spec.model_dump())
        assert roundtripped.benchmark_id == spec.benchmark_id
        assert roundtripped.dataset.id == spec.dataset.id
        assert roundtripped.driver.type == spec.driver.type


def test_chatbot_short_is_closed_loop() -> None:
    plugin = LLMInferencePlugin()
    spec = plugin.get_benchmark("llm.inference.chatbot-short")
    assert isinstance(spec, BenchmarkSpec)
    assert spec.driver.type == "closed_loop"
    assert spec.driver.concurrency == [4, 16, 64]
    assert spec.driver.duration_s == 120
    assert spec.slo_template == "llm.standard"
    assert spec.dataset.uri == "builtin://"


def test_long_context_is_open_loop() -> None:
    plugin = LLMInferencePlugin()
    spec = plugin.get_benchmark("llm.inference.long-context")
    assert isinstance(spec, BenchmarkSpec)
    assert spec.driver.type == "open_loop"
    assert spec.driver.rps == [0.5, 1, 2]
    assert spec.driver.duration_s == 300
    assert spec.slo_template == "llm.relaxed"
    assert spec.dataset.uri == "builtin://"


def test_missing_benchmark_raises_keyerror() -> None:
    plugin = LLMInferencePlugin()
    with pytest.raises(KeyError):
        plugin.get_benchmark("llm.inference.nonexistent")
