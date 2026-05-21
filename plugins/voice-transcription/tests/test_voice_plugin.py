"""End-to-end tests for the voice-transcription plugin.

The plugin's real-audio path (`audio_client.transcribe`) is mocked in these
tests — no live HTTP server is contacted. Tests cover the plugin contract,
validation, scoring aggregation, and the missing-WAV degradation path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from inferencebench_voice import (
    BenchmarkSpec,
    EngineKind,
    RunContext,
    VoiceTranscriptionPlugin,
)
from inferencebench_voice.audio_client import TranscriptionResult


# --------------------------------------------------------------------------- #
# Plugin contract                                                             #
# --------------------------------------------------------------------------- #
def test_plugin_metadata() -> None:
    plugin = VoiceTranscriptionPlugin()
    assert plugin.suite_id == "voice.transcription"
    assert plugin.version
    assert plugin.description


def test_plugin_lists_bundled_benchmarks() -> None:
    plugin = VoiceTranscriptionPlugin()
    specs = plugin.list_benchmarks()
    assert len(specs) >= 5
    ids = {s.benchmark_id for s in specs}
    assert {
        "voice.transcription.fleurs-mini",
        "voice.transcription.long-form",
        "voice.transcription.code-switched-mini",
        "voice.transcription.accented-mini",
        "voice.transcription.librispeech-clean-mini",
    }.issubset(ids)


def test_get_benchmark_code_switched_mini_resolves() -> None:
    plugin = VoiceTranscriptionPlugin()
    spec = plugin.get_benchmark("voice.transcription.code-switched-mini")
    assert isinstance(spec, BenchmarkSpec)
    assert spec.scoring == "wer"
    assert spec.dataset.path == "code-switched-mini.jsonl"


def test_get_benchmark_accented_mini_resolves() -> None:
    plugin = VoiceTranscriptionPlugin()
    spec = plugin.get_benchmark("voice.transcription.accented-mini")
    assert isinstance(spec, BenchmarkSpec)
    assert spec.scoring == "wer"
    assert spec.dataset.path == "accented-mini.jsonl"


def test_get_benchmark_librispeech_clean_mini_resolves() -> None:
    plugin = VoiceTranscriptionPlugin()
    spec = plugin.get_benchmark("voice.transcription.librispeech-clean-mini")
    assert isinstance(spec, BenchmarkSpec)
    assert spec.scoring == "wer"
    assert spec.dataset.path == "librispeech-clean-mini.jsonl"


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
# End-to-end run with mocked audio client                                     #
# --------------------------------------------------------------------------- #
def _patch_transcribe_returning(
    monkeypatch: pytest.MonkeyPatch,
    text_for_reference: dict[str, str] | None = None,
    *,
    constant_text: str | None = None,
    total_ms: float = 123.0,
) -> list[Path]:
    """Patch the plugin's audio-call seam.

    Returns the list of audio paths the patched function was asked to transcribe
    (in call order) so tests can assert which fixture rows were exercised.
    """
    seen: list[Path] = []

    def _fake(
        self: Any,
        audio_path: Path,
        *,
        base_url: str,
        model: str,
        api_key: str,
    ) -> TranscriptionResult:
        seen.append(audio_path)
        text = constant_text if constant_text is not None else ""
        if text_for_reference is not None:
            # Match by filename — tests register lookup tables keyed by WAV name.
            text = text_for_reference.get(audio_path.name, text)
        return TranscriptionResult(
            text=text,
            total_ms=total_ms,
            ttft_ms=total_ms,
            tokens_out=len(text.split()),
        )

    monkeypatch.setattr(
        VoiceTranscriptionPlugin,
        "_invoke_transcribe",
        _fake,
        raising=True,
    )
    return seen


def test_run_with_mocked_transcribe_returning_reference_yields_zero_wer(
    make_run_context: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end: stub returns the reference verbatim -> WER = 0."""
    plugin = VoiceTranscriptionPlugin()
    spec = plugin.get_benchmark("voice.transcription.fleurs-mini")

    # Map each fixture WAV to its reference string.
    references = {
        "fleurs-001.wav": "the quick brown fox jumps over the lazy dog",
        "fleurs-002.wav": "she sells seashells by the seashore",
        "fleurs-003.wav": "how much wood would a woodchuck chuck",
        "fleurs-004.wav": "peter piper picked a peck of pickled peppers",
        "fleurs-005.wav": "the rain in spain falls mainly on the plain",
    }
    seen = _patch_transcribe_returning(monkeypatch, references, total_ms=50.0)

    envelope = plugin.run(spec, make_run_context())

    assert len(seen) == 5
    assert envelope.signature is not None
    wer_mean = envelope.metrics.get("wer_mean")
    assert wer_mean == pytest.approx(0.0, abs=1e-12)
    assert envelope.metrics["wer_p50"] == pytest.approx(0.0, abs=1e-12)
    assert envelope.metrics["wer_p95"] == pytest.approx(0.0, abs=1e-12)
    assert envelope.metrics["ok_rate"] == 1.0
    assert envelope.metrics["n_samples"] == 5.0
    assert envelope.metrics["audio_path_resolved_count"] == 5.0
    assert envelope.metrics["total_p50_ms"] == pytest.approx(50.0)


def test_run_with_corrupted_hypothesis_matches_handcomputed_wer(
    make_run_context: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stub drops the last word of each reference -> WER == 1/n_words per row."""
    plugin = VoiceTranscriptionPlugin()
    spec = plugin.get_benchmark("voice.transcription.fleurs-mini")

    refs = {
        "fleurs-001.wav": "the quick brown fox jumps over the lazy dog",
        "fleurs-002.wav": "she sells seashells by the seashore",
        "fleurs-003.wav": "how much wood would a woodchuck chuck",
        "fleurs-004.wav": "peter piper picked a peck of pickled peppers",
        "fleurs-005.wav": "the rain in spain falls mainly on the plain",
    }
    # Drop the last word — yields exactly one deletion -> WER = 1/n_words.
    corrupted = {name: " ".join(text.split()[:-1]) for name, text in refs.items()}
    _patch_transcribe_returning(monkeypatch, corrupted)

    envelope = plugin.run(spec, make_run_context())
    expected = sum(1 / 9 + 1 / 6 + 1 / 7 + 1 / 8 + 1 / 9 for _ in [0]) / 5
    assert envelope.metrics["wer_mean"] == pytest.approx(expected, rel=1e-9)


def test_run_librispeech_mini_with_whisper_style_hypothesis_yields_zero_wer(
    make_run_context: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mock returns lowercase, punctuated, Whisper-flavoured hypotheses against
    the all-caps LibriSpeech references — the normalizer in the WER scorer
    must absorb the difference and report 0% WER."""
    plugin = VoiceTranscriptionPlugin()
    spec = plugin.get_benchmark("voice.transcription.librispeech-clean-mini")

    # Hypotheses styled exactly the way Whisper / OpenAI audio renders them:
    # sentence-case, trailing period, Unicode right-single-quote for "doesn't".
    whisper_style = {
        "ls-001.wav": "Inquired Shaggy, in the metal forest.",
        "ls-002.wav": "As for etchings, they are of two kinds, British and foreign.",
        "ls-003.wav": "He eats and sleeps very steadily, replied the new king.",
        "ls-004.wav": "I hope he doesn’t work too hard, said Shaggy.",  # noqa: RUF001 — U+2019 is intentional, mirrors Whisper output
        "ls-005.wav": "Not exactly, returned Kaliko.",
    }
    seen = _patch_transcribe_returning(monkeypatch, whisper_style, total_ms=120.0)

    envelope = plugin.run(spec, make_run_context())

    assert len(seen) == 5
    assert envelope.signature is not None
    assert envelope.metrics["n_samples"] == 5.0
    assert envelope.metrics["ok_rate"] == 1.0
    assert envelope.metrics["wer_mean"] == pytest.approx(0.0, abs=1e-12)


def test_run_writes_samples_jsonl_alongside_envelope(
    make_run_context: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The diagnostic samples-<ts>.jsonl is written to output_dir."""
    plugin = VoiceTranscriptionPlugin()
    spec = plugin.get_benchmark("voice.transcription.fleurs-mini")
    ctx = make_run_context()
    _patch_transcribe_returning(monkeypatch, constant_text="ok")

    plugin.run(spec, ctx)
    samples_files = list(ctx.output_dir.glob("samples-*.jsonl"))
    assert len(samples_files) == 1
    lines = samples_files[0].read_text(encoding="utf-8").splitlines()
    assert len(lines) == 5  # one per fixture row


def test_run_missing_wav_skips_row_and_drops_resolved_count(
    make_run_context: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A fixture row that points at a missing WAV is recorded as failed.

    The envelope's ``audio_path_resolved_count`` reflects the drop and
    ``n_ok`` excludes the failed row.
    """
    # Build a temporary plugin instance with a synthesized fixture that
    # points at one missing WAV and one real one.
    plugin = VoiceTranscriptionPlugin()
    datasets_dir = Path(plugin._datasets_dir())
    custom_fixture = datasets_dir / "_test_missing_wav.jsonl"
    custom_fixture.write_text(
        '{"audio_path": "audio/fleurs-001.wav", "reference": "hello", "duration_s": 0.3}\n'
        '{"audio_path": "audio/does_not_exist.wav", "reference": "world", "duration_s": 0.3}\n',
        encoding="utf-8",
    )
    try:
        spec = BenchmarkSpec(
            benchmark_id="voice.transcription.test-missing",
            suite_version="0.0.1",
            dataset={"id": "test-missing", "path": "_test_missing_wav.jsonl"},
            scoring="wer",
        )
        _patch_transcribe_returning(monkeypatch, constant_text="hello")

        envelope = plugin.run(spec, make_run_context())
        assert envelope.metrics["n_samples"] == 2.0
        assert envelope.metrics["n_ok"] == 1.0
        assert envelope.metrics["audio_path_resolved_count"] == 1.0
        # Only the resolved row contributed to WER.
        assert envelope.metrics["wer_mean"] == pytest.approx(0.0, abs=1e-12)
    finally:
        custom_fixture.unlink(missing_ok=True)
