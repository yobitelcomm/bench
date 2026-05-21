"""Tests for the vision-understanding plugin scaffold + scoring pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from inferencebench.cli import app
from inferencebench.envelope import generate_dev_keypair
from inferencebench_vision import (
    EXPECTED_METRICS,
    BenchmarkSpec,
    EngineKind,
    RunContext,
    VisionUnderstandingPlugin,
)
from inferencebench_vision.multimodal_client import (
    build_multimodal_messages,
)


# --------------------------------------------------------------------------- #
# Plugin contract                                                             #
# --------------------------------------------------------------------------- #
def test_plugin_metadata() -> None:
    plugin = VisionUnderstandingPlugin()
    assert plugin.suite_id == "vision.understanding"
    assert plugin.version == "0.0.2"
    assert plugin.description


def test_plugin_lists_bundled_benchmarks() -> None:
    plugin = VisionUnderstandingPlugin()
    specs = plugin.list_benchmarks()
    assert len(specs) >= 2
    ids = {s.benchmark_id for s in specs}
    assert {
        "vision.understanding.ocr-mini",
        "vision.understanding.chart-qa-mini",
    }.issubset(ids)


def test_get_benchmark_ocr_mini_resolves() -> None:
    plugin = VisionUnderstandingPlugin()
    spec = plugin.get_benchmark("vision.understanding.ocr-mini")
    assert isinstance(spec, BenchmarkSpec)
    assert spec.modality == "vision"
    assert spec.kind == "understanding"
    assert spec.scoring == "substring_match"
    assert spec.dataset.path == "ocr-mini.jsonl"


def test_get_benchmark_chart_qa_mini_uses_exact_match() -> None:
    plugin = VisionUnderstandingPlugin()
    spec = plugin.get_benchmark("vision.understanding.chart-qa-mini")
    assert spec.scoring == "exact_match"


def test_get_benchmark_missing_id_raises_keyerror() -> None:
    plugin = VisionUnderstandingPlugin()
    with pytest.raises(KeyError):
        plugin.get_benchmark("nonexistent.benchmark")


def test_expected_metrics_includes_accuracy_band() -> None:
    assert "accuracy" in EXPECTED_METRICS
    assert "accuracy_p05" in EXPECTED_METRICS
    assert "accuracy_p50" in EXPECTED_METRICS
    assert "accuracy_p95" in EXPECTED_METRICS
    assert "tokens_out_total" in EXPECTED_METRICS


# --------------------------------------------------------------------------- #
# Validation                                                                  #
# --------------------------------------------------------------------------- #
def test_validate_warns_when_self_hosted_base_url_missing() -> None:
    plugin = VisionUnderstandingPlugin()
    spec = plugin.get_benchmark("vision.understanding.ocr-mini")
    ctx = RunContext(
        model_id="m",
        engine_kind=EngineKind.VLLM,
        base_url="",
        output_dir=Path("/tmp/bench"),
    )
    warnings = plugin.validate(spec, ctx)
    assert any("base_url" in w.lower() for w in warnings)


def test_validate_openai_engine_does_not_require_base_url() -> None:
    plugin = VisionUnderstandingPlugin()
    spec = plugin.get_benchmark("vision.understanding.ocr-mini")
    ctx = RunContext(
        model_id="openai/gpt-4o",
        engine_kind=EngineKind.OPENAI,
        base_url="",
        output_dir=Path("/tmp/bench"),
    )
    warnings = plugin.validate(spec, ctx)
    assert not any("base_url" in w.lower() for w in warnings)


def test_validate_anthropic_engine_does_not_require_base_url() -> None:
    plugin = VisionUnderstandingPlugin()
    spec = plugin.get_benchmark("vision.understanding.ocr-mini")
    ctx = RunContext(
        model_id="anthropic/claude-opus-4-7",
        engine_kind=EngineKind.ANTHROPIC,
        base_url="",
        output_dir=Path("/tmp/bench"),
    )
    warnings = plugin.validate(spec, ctx)
    assert not any("base_url" in w.lower() for w in warnings)


# --------------------------------------------------------------------------- #
# Multimodal payload shape                                                    #
# --------------------------------------------------------------------------- #
def test_build_multimodal_messages_shape() -> None:
    """Payload must be the OpenAI/Anthropic list-of-parts shape with a data URL."""
    image_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "inferencebench_vision"
        / "datasets"
        / "images"
        / "ocr-01.png"
    )
    messages = build_multimodal_messages(image_path, "What text?")
    assert len(messages) == 1
    msg = messages[0]
    assert msg["role"] == "user"
    content = msg["content"]
    assert isinstance(content, list)
    assert content[0] == {"type": "text", "text": "What text?"}
    assert content[1]["type"] == "image_url"
    url = content[1]["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")
    # Make sure the base64 segment is non-trivial.
    assert len(url) > len("data:image/png;base64,") + 100


# --------------------------------------------------------------------------- #
# End-to-end run (mocked client)                                              #
# --------------------------------------------------------------------------- #
def _ocr_answers() -> dict[str, str]:
    return {
        "images/ocr-01.png": "april 17",
        "images/ocr-02.png": "invoice 4421",
        "images/ocr-03.png": "total $89.50",
        "images/ocr-04.png": "order #7732",
        "images/ocr-05.png": "due may 03",
    }


def test_run_produces_signed_envelope_with_high_accuracy(
    make_mock_modelclient, tmp_path: Path
) -> None:
    """Mock returns the canonical answer for every image → accuracy = 1.0."""
    answers = _ocr_answers()

    def responder(image_path: Path, _question: str) -> str:
        # Reverse-lookup the relative path under datasets/images/.
        for rel, text in answers.items():
            if image_path.name == Path(rel).name:
                return text
        return ""

    make_mock_modelclient(responder)

    private_key_path = tmp_path / "cosign.key"
    generate_dev_keypair(private_key_path)

    plugin = VisionUnderstandingPlugin()
    spec = plugin.get_benchmark("vision.understanding.ocr-mini")
    ctx = RunContext(
        model_id="openai/mock-vlm",
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
    assert envelope.signature.method == "dev-key"
    assert envelope.signature.bundle

    acc = envelope.metrics.get("accuracy")
    assert acc is not None
    assert isinstance(acc, (int, float))
    assert float(acc) == 1.0
    assert envelope.metrics.get("ok_rate") == 1.0
    assert envelope.metrics.get("n_samples") == 5.0
    assert envelope.metrics.get("accuracy_p50") is not None


def test_run_partial_accuracy_with_mixed_answers(make_mock_modelclient, tmp_path: Path) -> None:
    """First 3 rows answered correctly, last 2 wrong → accuracy = 0.6."""
    answers = _ocr_answers()
    correct_files = ["ocr-01.png", "ocr-02.png", "ocr-03.png"]

    def responder(image_path: Path, _question: str) -> str:
        if image_path.name in correct_files:
            for rel, text in answers.items():
                if image_path.name == Path(rel).name:
                    return text
        return "definitely not a real answer xyz"

    make_mock_modelclient(responder)

    private_key_path = tmp_path / "cosign.key"
    generate_dev_keypair(private_key_path)

    plugin = VisionUnderstandingPlugin()
    spec = plugin.get_benchmark("vision.understanding.ocr-mini")
    ctx = RunContext(
        model_id="openai/mock-vlm",
        engine_kind=EngineKind.OPENAI,
        output_dir=tmp_path / "out",
        extra={
            "signing_mode": "dev",
            "dev_key_path": str(private_key_path),
        },
    )

    envelope = plugin.run(spec, ctx)
    acc = envelope.metrics.get("accuracy")
    assert acc is not None
    assert float(acc) == pytest.approx(0.6)
    assert envelope.metrics.get("ok_rate") == 1.0


def test_run_writes_samples_jsonl(make_mock_modelclient, tmp_path: Path) -> None:
    make_mock_modelclient(lambda _image, _question: "april 17")

    private_key_path = tmp_path / "cosign.key"
    generate_dev_keypair(private_key_path)

    plugin = VisionUnderstandingPlugin()
    spec = plugin.get_benchmark("vision.understanding.ocr-mini")
    out_dir = tmp_path / "out"
    ctx = RunContext(
        model_id="openai/mock-vlm",
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
    assert len(lines) == 5  # one per fixture row


def test_run_envelope_accepted_by_bench_summary(make_mock_modelclient, tmp_path: Path) -> None:
    """End-to-end: write the envelope JSON to a tmp dir, run ``bench summary``."""
    answers = _ocr_answers()

    def responder(image_path: Path, _question: str) -> str:
        for rel, text in answers.items():
            if image_path.name == Path(rel).name:
                return text
        return ""

    make_mock_modelclient(responder)

    private_key_path = tmp_path / "cosign.key"
    generate_dev_keypair(private_key_path)

    plugin = VisionUnderstandingPlugin()
    spec = plugin.get_benchmark("vision.understanding.ocr-mini")
    env_dir = tmp_path / "envelopes"
    env_dir.mkdir()
    ctx = RunContext(
        model_id="openai/mock-vlm",
        engine_kind=EngineKind.OPENAI,
        output_dir=tmp_path / "out",
        extra={
            "signing_mode": "dev",
            "dev_key_path": str(private_key_path),
        },
    )

    envelope = plugin.run(spec, ctx)
    env_path = env_dir / "vision-run.json"
    env_path.write_text(
        json.dumps(envelope.model_dump(mode="json"), indent=2),
        encoding="utf-8",
    )

    runner = CliRunner(env={"COLUMNS": "240"})
    result = runner.invoke(app, ["summary", str(env_dir)])
    assert result.exit_code == 0, result.output
    assert "vision.understanding.ocr-mini" in result.output
