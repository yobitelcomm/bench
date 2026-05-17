"""InferenceBench voice-transcription plugin."""

from inferencebench_voice.plugin import VoiceTranscriptionPlugin
from inferencebench_voice.schemas import BenchmarkSpec, EngineKind, RunContext

__all__ = ["BenchmarkSpec", "EngineKind", "RunContext", "VoiceTranscriptionPlugin"]
