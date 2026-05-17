"""Tests for the voice-transcription plugin scaffold + scoring pipeline."""

from __future__ import annotations

from pathlib import Path

import pytest

from inferencebench_voice import (
    BenchmarkSpec,
    EngineKind,
    RunContext,
    VoiceTranscriptionPlugin,
)


# --------------------------------------------------------------------------- #
# Plugin contract                                                             #
# --------------------------------------------------------------------------- #
def test_plugin_metadata() -> None:
    plugin = VoiceTranscriptionPlugin()
    assert plugin.suite_id == "voice.transcription"
    assert plugin.version
    assert plugin.description


def test_plugin_lists_two_bundled_benchmarks() -> None:
    plugin = VoiceTranscriptionPlugin()
    specs = plugin.list_benchmarks()
    assert len(specs) == 2
    ids = {s.benchmark_id for s in specs}
    assert ids == {"voice.transcription.fleurs-mini", "voice.transcription.long-form"}


def test_plugin_get_benchmark_fleurs_mini() -> None:
    plugin = VoiceTranscriptionPlugin()
    spec = plugin.get_benchmark("voice.transcription.fleurs-mini")
    assert isinstance(spec, BenchmarkSpec)
    assert spec.modality == "voice"
    assert spec.kind == "transcription"
    assert spec.scoring == "wer"
    assert spec.dataset.path == "fleurs-mini.jsonl"


def test_plugin_get_benchmark_missing_id_raises_keyerror() -> None:
    plugin = VoiceTranscriptionPlugin()
    with pytest.raises(KeyError):
        plugin.get_benchmark("nonexistent.benchmark")


# --------------------------------------------------------------------------- #
# Validation                                                                  #
# --------------------------------------------------------------------------- #
def test_validate_warns_when_self_hosted_base_url_missing() -> None:
    plugin = VoiceTranscriptionPlugin()
    spec = plugin.get_benchmark("voice.transcription.fleurs-mini")
    ctx = RunContext(
        model_id="m",
        engine_kind=EngineKind.WHISPER_HTTP,
        base_url="",
        output_dir=Path("/tmp/bench"),
    )
    warnings = plugin.validate(spec, ctx)
    assert any("base_url" in w.lower() for w in warnings)


def test_validate_provider_hosted_engine_does_not_require_base_url() -> None:
    plugin = VoiceTranscriptionPlugin()
    spec = plugin.get_benchmark("voice.transcription.fleurs-mini")
    ctx = RunContext(
        model_id="openai/whisper-1",
        engine_kind=EngineKind.OPENAI,
        base_url="",
        output_dir=Path("/tmp/bench"),
    )
    warnings = plugin.validate(spec, ctx)
    assert not any("base_url" in w.lower() for w in warnings)


# --------------------------------------------------------------------------- #
# End-to-end run (no real audio invocation)                                   #
# --------------------------------------------------------------------------- #
def test_run_produces_signed_envelope_with_expected_wer(make_run_context) -> None:
    """Skeleton substitutes the last word -> WER is the (1/n) average across rows."""
    plugin = VoiceTranscriptionPlugin()
    spec = plugin.get_benchmark("voice.transcription.fleurs-mini")
    ctx = make_run_context()

    envelope = plugin.run(spec, ctx)

    # Signature is real
    assert envelope.signature is not None
    assert envelope.signature.method == "dev-key"
    assert envelope.signature.bundle  # non-empty base64 blob

    wer_mean = envelope.metrics.get("wer_mean")
    assert wer_mean is not None
    assert isinstance(wer_mean, (int, float))
    # 5 fleurs-mini rows: 9, 6, 7, 8, 9 words -> WER 1/n each.
    expected = sum(x for x in (1 / 9, 1 / 6, 1 / 7, 1 / 8, 1 / 9)) / 5
    assert float(wer_mean) == pytest.approx(expected, rel=1e-9)

    # Supplementary metrics are present.
    assert envelope.metrics.get("wer_p50") is not None
    assert envelope.metrics.get("wer_p95") is not None
    assert envelope.metrics.get("total_audio_duration_s") == pytest.approx(14.8)
    assert envelope.metrics.get("total_p50_ms") is not None
    assert envelope.metrics.get("ok_rate") == 1.0
    assert envelope.metrics.get("n_samples") == 5.0


def test_run_long_form_envelope_has_lower_wer(make_run_context) -> None:
    """Long references -> stub WER is ~1/(40-100 words) -> smaller WER mean."""
    plugin = VoiceTranscriptionPlugin()
    spec = plugin.get_benchmark("voice.transcription.long-form")
    ctx = make_run_context()
    envelope = plugin.run(spec, ctx)

    wer_mean = envelope.metrics.get("wer_mean")
    assert wer_mean is not None
    # All references are >40 words → WER per row <= 1/40 = 0.025.
    assert 0.0 < float(wer_mean) <= 0.025
    assert envelope.metrics.get("n_samples") == 3.0


def test_run_writes_samples_jsonl_alongside_envelope(make_run_context) -> None:
    """The diagnostic samples-<ts>.jsonl is written to output_dir."""
    plugin = VoiceTranscriptionPlugin()
    spec = plugin.get_benchmark("voice.transcription.fleurs-mini")
    ctx = make_run_context()

    plugin.run(spec, ctx)
    samples_files = list(ctx.output_dir.glob("samples-*.jsonl"))
    assert len(samples_files) == 1
    lines = samples_files[0].read_text(encoding="utf-8").splitlines()
    assert len(lines) == 5  # one per fixture row
