"""One-shot generator for the voice-transcription plugin's bundled WAV fixtures.

Produces 20 tiny 16 kHz mono PCM WAV files (sine tones, ~0.4 s each) that the
plugin can ship to a Whisper-compatible HTTP endpoint. The files are NOT real
speech — a Whisper server will return ``"..."`` or random garbage for them —
but the request shape is exercised end-to-end. Maintainers swap in real WAVs
for production runs.

Total disk impact is held below ~200 KB by keeping each clip short (0.3-0.5 s
sine wave at 16 kHz mono int16 = ~10-16 KB per file).

Run once from the repo root::

    uv run python tools/generate_voice_wavs.py

The generated WAVs are checked in alongside the fixture JSONLs.
"""

from __future__ import annotations

import math
import struct
import wave
from pathlib import Path

# 16 kHz mono PCM — the Whisper canonical input shape.
SAMPLE_RATE = 16_000
N_CHANNELS = 1
SAMPLE_WIDTH = 2  # int16
AMPLITUDE = 16_000  # ~half full-scale, leaves headroom for any clipping

# Each entry: (filename, frequency_hz, duration_seconds).
# Durations chosen so total payload stays well under 200 KB.
# 0.4 s @ 16 kHz mono int16 = 12.8 KB per file x 20 files = ~256 KB raw;
# we shave to 0.3 s for the 10 "speech-like" clips to stay under budget.
CLIPS: tuple[tuple[str, float, float], ...] = (
    # fleurs-mini (5)
    ("fleurs-001.wav", 440.0, 0.30),
    ("fleurs-002.wav", 466.16, 0.30),
    ("fleurs-003.wav", 493.88, 0.30),
    ("fleurs-004.wav", 523.25, 0.30),
    ("fleurs-005.wav", 554.37, 0.30),
    # long-form (3 referenced + 2 spare for symmetry with the task brief)
    ("long-001.wav", 587.33, 0.30),
    ("long-002.wav", 622.25, 0.30),
    ("long-003.wav", 659.25, 0.30),
    ("long-004.wav", 698.46, 0.30),
    ("long-005.wav", 739.99, 0.30),
    # code-switched-mini (5)
    ("cs-01.wav", 783.99, 0.30),
    ("cs-02.wav", 830.61, 0.30),
    ("cs-03.wav", 880.0, 0.30),
    ("cs-04.wav", 932.33, 0.30),
    ("cs-05.wav", 987.77, 0.30),
    # accented-mini (5)
    ("ac-01.wav", 1046.5, 0.30),
    ("ac-02.wav", 1108.73, 0.30),
    ("ac-03.wav", 1174.66, 0.30),
    ("ac-04.wav", 1244.51, 0.30),
    ("ac-05.wav", 1318.51, 0.30),
)


def _sine_pcm16_bytes(freq_hz: float, duration_s: float) -> bytes:
    """Render ``duration_s`` of a pure sine at ``freq_hz`` as packed int16 LE."""
    n_frames = int(SAMPLE_RATE * duration_s)
    two_pi_f = 2.0 * math.pi * freq_hz
    inv_sr = 1.0 / SAMPLE_RATE
    samples = bytearray(n_frames * SAMPLE_WIDTH)
    pack_into = struct.pack_into
    for i in range(n_frames):
        val = int(AMPLITUDE * math.sin(two_pi_f * i * inv_sr))
        pack_into("<h", samples, i * SAMPLE_WIDTH, val)
    return bytes(samples)


def _write_wav(path: Path, pcm: bytes) -> None:
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(N_CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm)


def main() -> None:
    """Generate every clip in :data:`CLIPS` into the plugin's ``datasets/audio/``."""
    out_dir = (
        Path(__file__).resolve().parent.parent
        / "plugins"
        / "voice-transcription"
        / "src"
        / "inferencebench_voice"
        / "datasets"
        / "audio"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    total_bytes = 0
    for name, freq, dur in CLIPS:
        pcm = _sine_pcm16_bytes(freq, dur)
        path = out_dir / name
        _write_wav(path, pcm)
        size = path.stat().st_size
        total_bytes += size
        print(f"wrote {path.relative_to(out_dir.parents[3])}  ({size} bytes)")

    print(f"\ntotal: {len(CLIPS)} files, {total_bytes} bytes")
    if total_bytes >= 200_000:
        msg = f"bundled WAVs exceed 200 KB budget: {total_bytes} bytes"
        raise SystemExit(msg)


if __name__ == "__main__":
    main()
