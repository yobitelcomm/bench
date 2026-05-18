"""Tests for the bundled synthetic WAV fixtures.

Every bundled WAV must:
    * Live under ``datasets/audio/`` inside the installed package.
    * Open with :mod:`wave` (i.e. be a valid RIFF/WAVE PCM container).
    * Be 16 kHz mono PCM — the Whisper canonical input shape.

The total payload is held under 200 KB so the wheel stays small.
"""

from __future__ import annotations

import wave
from pathlib import Path

import pytest

import inferencebench_voice

_AUDIO_DIR = Path(inferencebench_voice.__file__).parent / "datasets" / "audio"

_EXPECTED = (
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


@pytest.mark.parametrize("name", _EXPECTED)
def test_bundled_wav_exists_and_parses(name: str) -> None:
    path = _AUDIO_DIR / name
    assert path.exists(), f"missing: {path}"
    # Must open cleanly as PCM WAV.
    with wave.open(str(path), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getframerate() == 16_000
        assert wf.getsampwidth() == 2  # int16


def test_bundled_wavs_total_size_under_200kb() -> None:
    sizes = [(_AUDIO_DIR / name).stat().st_size for name in _EXPECTED]
    total = sum(sizes)
    # Hard cap from the task brief.
    assert total < 200_000, f"bundled WAVs total {total} bytes (> 200 KB)"


def test_bundled_wav_count_is_twenty() -> None:
    wavs = sorted(_AUDIO_DIR.glob("*.wav"))
    assert len(wavs) == 20
