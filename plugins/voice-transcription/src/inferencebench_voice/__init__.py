"""InferenceBench voice-transcription plugin."""

from inferencebench_voice.plugin import EXPECTED_METRICS, VoiceTranscriptionPlugin
from inferencebench_voice.schemas import BenchmarkSpec, EngineKind, RunContext

__all__ = [
    "EXPECTED_METRICS",
    "BenchmarkSpec",
    "EngineKind",
    "RunContext",
    "VoiceTranscriptionPlugin",
]
