"""Tests for the bundled WAV fixtures.

Every bundled WAV must:
    * Live under ``datasets/audio/`` inside the installed package.
    * Open with :mod:`wave` (i.e. be a valid RIFF/WAVE PCM container).
    * Be 16 kHz mono PCM — the Whisper canonical input shape.

Two families ship in-tree:
    * 20 synthetic sine-tone WAVs (~10 KB each) covering fleurs / long / cs / ac
      mini-suites — exercise the wire format only; Whisper returns garbage on them.
    * 5 real LibriSpeech test-clean utterances (~120 KB each), CC BY 4.0,
      sourced from `hf-internal-testing/librispeech_asr_dummy`. These are
      real speech and produce realistic WER against a Whisper engine.

Total payload is held under 1 MB to keep the wheel tractable.
"""

from __future__ import annotations

import wave
from pathlib import Path

import pytest

import inferencebench_voice

_AUDIO_DIR = Path(inferencebench_voice.__file__).parent / "datasets" / "audio"

_SYNTHETIC = (
    "fleurs-001.wav",
    "fleurs-002.wav",
    "fleurs-003.wav",
    "fleurs-004.wav",
    "fleurs-005.wav",
    "long-001.wav",
    "long-002.wav",
    "long-003.wav",
    "long-004.wav",
    "long-005.wav",
    "cs-01.wav",
    "cs-02.wav",
    "cs-03.wav",
    "cs-04.wav",
    "cs-05.wav",
    "ac-01.wav",
    "ac-02.wav",
    "ac-03.wav",
    "ac-04.wav",
    "ac-05.wav",
)

_LIBRISPEECH = (
    "ls-001.wav",
    "ls-002.wav",
    "ls-003.wav",
    "ls-004.wav",
    "ls-005.wav",
)

_EXPECTED = _SYNTHETIC + _LIBRISPEECH


@pytest.mark.parametrize("name", _EXPECTED)
def test_bundled_wav_exists_and_parses(name: str) -> None:
    path = _AUDIO_DIR / name
    assert path.exists(), f"missing: {path}"
    with wave.open(str(path), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getframerate() == 16_000
        assert wf.getsampwidth() == 2  # int16


def test_synthetic_wavs_total_size_under_200kb() -> None:
    sizes = [(_AUDIO_DIR / name).stat().st_size for name in _SYNTHETIC]
    total = sum(sizes)
    assert total < 200_000, f"synthetic WAVs total {total} bytes (> 200 KB)"


def test_librispeech_wavs_total_size_under_1mb() -> None:
    sizes = [(_AUDIO_DIR / name).stat().st_size for name in _LIBRISPEECH]
    total = sum(sizes)
    assert total < 1_000_000, f"LibriSpeech WAVs total {total} bytes (> 1 MB)"


def test_librispeech_wavs_are_real_speech_not_silent() -> None:
    # Sanity check: real LibriSpeech utterances must contain non-trivial audio
    # energy. A silent file would be ~zeroes; sine tones average around ~half-scale.
    for name in _LIBRISPEECH:
        path = _AUDIO_DIR / name
        with wave.open(str(path), "rb") as wf:
            n_frames = wf.getnframes()
            assert n_frames > 16_000, f"{name} too short for real speech ({n_frames} frames)"
            raw = wf.readframes(n_frames)
        # Cheap energy proxy: at least 5% of bytes are non-zero in the int16 stream.
        nonzero_ratio = sum(b != 0 for b in raw) / max(len(raw), 1)
        assert nonzero_ratio > 0.05, f"{name} looks silent (nonzero ratio={nonzero_ratio:.3f})"


def test_bundled_wav_count_is_twenty_five() -> None:
    wavs = sorted(_AUDIO_DIR.glob("*.wav"))
    assert len(wavs) == 25
