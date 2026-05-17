"""Tests for the llm-quality plugin scaffold + scoring pipeline."""

from __future__ import annotations

from pathlib import Path

import pytest

from inferencebench.envelope import generate_dev_keypair
from inferencebench_quality import (
    BenchmarkSpec,
    EngineKind,
    LLMQualityPlugin,
    RunContext,
)


# --------------------------------------------------------------------------- #
# Plugin contract                                                             #
# --------------------------------------------------------------------------- #
def test_plugin_metadata() -> None:
    plugin = LLMQualityPlugin()
    assert plugin.suite_id == "llm.quality"
    assert plugin.version
    assert plugin.description


def test_plugin_lists_two_bundled_benchmarks() -> None:
    plugin = LLMQualityPlugin()
    specs = plugin.list_benchmarks()
    assert len(specs) == 2
    ids = {s.benchmark_id for s in specs}
    assert ids == {"llm.quality.factual-mini", "llm.quality.reasoning-mini"}


def test_plugin_get_benchmark_factual_mini() -> None:
    plugin = LLMQualityPlugin()
    spec = plugin.get_benchmark("llm.quality.factual-mini")
    assert isinstance(spec, BenchmarkSpec)
    assert spec.modality == "llm"
    assert spec.kind == "quality"
    assert spec.scoring == "substring_match"
    assert spec.dataset.path == "factual-mini.jsonl"


def test_plugin_get_benchmark_reasoning_mini_uses_exact_match() -> None:
    plugin = LLMQualityPlugin()
    spec = plugin.get_benchmark("llm.quality.reasoning-mini")
    assert spec.scoring == "exact_match"


def test_plugin_get_benchmark_missing_id_raises_keyerror() -> None:
    plugin = LLMQualityPlugin()
    with pytest.raises(KeyError):
        plugin.get_benchmark("nonexistent.benchmark")


# --------------------------------------------------------------------------- #
# Validation                                                                  #
# --------------------------------------------------------------------------- #
def test_validate_warns_when_self_hosted_base_url_missing() -> None:
    plugin = LLMQualityPlugin()
    spec = plugin.get_benchmark("llm.quality.factual-mini")
    ctx = RunContext(
        model_id="m",
        engine_kind=EngineKind.VLLM,
        base_url="",
        output_dir=Path("/tmp/bench"),
    )
    warnings = plugin.validate(spec, ctx)
    assert any("base_url" in w.lower() for w in warnings)


def test_validate_openai_engine_does_not_require_base_url() -> None:
    plugin = LLMQualityPlugin()
    spec = plugin.get_benchmark("llm.quality.factual-mini")
    ctx = RunContext(
        model_id="openai/gpt-4o-mini",
        engine_kind=EngineKind.OPENAI,
        base_url="",
        output_dir=Path("/tmp/bench"),
    )
    warnings = plugin.validate(spec, ctx)
    assert not any("base_url" in w.lower() for w in warnings)


# --------------------------------------------------------------------------- #
# End-to-end run (mocked client)                                              #
# --------------------------------------------------------------------------- #
def _expected_answers() -> dict[str, str]:
    """The substring-match ground truth from the bundled factual-mini fixture."""
    return {
        "What is the capital of France?": "Paris",
        "What is the capital of Japan?": "Tokyo",
        "Who wrote the play 'Hamlet'?": "Shakespeare",
        "What planet is known as the Red Planet?": "Mars",
        "What is the chemical symbol for gold?": "Au",
        "Which ocean is the largest by area?": "Pacific",
        "In what year did humans first land on the Moon?": "1969",
        "What is the largest mammal on Earth?": "blue whale",
        "What gas do plants primarily absorb for photosynthesis?": "carbon dioxide",
        "Who painted the Mona Lisa?": "Leonardo da Vinci",
    }


def test_run_produces_signed_envelope_with_high_accuracy(
    make_mock_modelclient, tmp_path: Path
) -> None:
    """Mock returns the canonical answer to every question → accuracy = 1.0."""
    answers = _expected_answers()

    def responder(prompt: str) -> str:
        return answers.get(prompt, "")

    make_mock_modelclient(responder)

    private_key_path = tmp_path / "cosign.key"
    generate_dev_keypair(private_key_path)

    plugin = LLMQualityPlugin()
    spec = plugin.get_benchmark("llm.quality.factual-mini")
    ctx = RunContext(
        model_id="openai/mock-model",
        model_revision="abc1234",
        engine_kind=EngineKind.OPENAI,
        output_dir=tmp_path / "out",
        extra={
            "signing_mode": "dev",
            "dev_key_path": str(private_key_path),
        },
    )

    envelope = plugin.run(spec, ctx)

    # Signature is real
    assert envelope.signature is not None
    assert envelope.signature.method == "dev-key"
    assert envelope.signature.bundle  # non-empty base64 blob

    # Accuracy is between 0 and 1
    acc = envelope.metrics.get("accuracy")
    assert acc is not None
    assert isinstance(acc, (int, float))
    assert 0.0 <= float(acc) <= 1.0
    # With the right answers, substring-match scores every row.
    assert float(acc) == 1.0

    # The expected supplementary metrics are present.
    assert envelope.metrics.get("accuracy_p50") is not None
    assert envelope.metrics.get("ok_rate") == 1.0
    assert envelope.metrics.get("n_samples") == 10.0


def test_run_with_wrong_answers_yields_zero_accuracy(
    make_mock_modelclient, tmp_path: Path
) -> None:
    """Mock always returns nonsense → accuracy = 0.0 but envelope still valid."""
    make_mock_modelclient(lambda _prompt: "definitely not a real answer xyz")

    private_key_path = tmp_path / "cosign.key"
    generate_dev_keypair(private_key_path)

    plugin = LLMQualityPlugin()
    spec = plugin.get_benchmark("llm.quality.factual-mini")
    ctx = RunContext(
        model_id="openai/mock-model",
        model_revision="abc1234",
        engine_kind=EngineKind.OPENAI,
        output_dir=tmp_path / "out",
        extra={
            "signing_mode": "dev",
            "dev_key_path": str(private_key_path),
        },
    )

    envelope = plugin.run(spec, ctx)
    assert envelope.signature is not None
    acc = envelope.metrics.get("accuracy")
    assert acc is not None
    assert float(acc) == 0.0
    assert envelope.metrics.get("ok_rate") == 1.0  # all calls succeeded, just wrong


def test_run_writes_samples_jsonl_alongside_envelope(
    make_mock_modelclient, tmp_path: Path
) -> None:
    """The diagnostic samples-<ts>.jsonl is written to output_dir."""
    make_mock_modelclient(lambda _prompt: "Paris")

    private_key_path = tmp_path / "cosign.key"
    generate_dev_keypair(private_key_path)

    plugin = LLMQualityPlugin()
    spec = plugin.get_benchmark("llm.quality.factual-mini")
    out_dir = tmp_path / "out"
    ctx = RunContext(
        model_id="openai/mock-model",
        engine_kind=EngineKind.OPENAI,
        output_dir=out_dir,
        extra={
            "signing_mode": "dev",
            "dev_key_path": str(private_key_path),
        },
    )

    plugin.run(spec, ctx)
    samples_files = list(out_dir.glob("samples-*.jsonl"))
    assert len(samples_files) == 1
    lines = samples_files[0].read_text(encoding="utf-8").splitlines()
    assert len(lines) == 10  # one per fixture row
