"""Pydantic schemas for voice-transcription benchmark specs + run context."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class EngineKind(StrEnum):
    """Engines this plugin can drive.

    The three production-grade paths a user actually cares about for ASR:
    a self-hosted OpenAI-compatible audio server (faster-whisper-server is
    the canonical implementation), OpenAI's provider-hosted audio endpoint,
    and Cohere's transcription API.
    """

    WHISPER_HTTP = "whisper-http"
    OPENAI = "openai"
    COHERE = "cohere"


class DatasetConfig(BaseModel):
    """Dataset under evaluation.

    Fixture is a JSONL with one ``{"audio_path", "reference", "duration_s"}``
    row per line. The audio path is a stub for the skeleton — see plugin docs.
    """

    model_config = ConfigDict(extra="forbid")
    id: Annotated[str, Field(min_length=1)]
    path: Annotated[
        str,
        Field(
            min_length=1,
            description=("Path to the fixture JSONL relative to the plugin's datasets/ directory."),
        ),
    ]


class WarmupConfig(BaseModel):
    """Warmup parameters.

    Transcription runs are per-utterance and order-independent, so the
    default is zero discarded runs. Surfaced for future JIT-warmup of
    server-side weights.
    """

    model_config = ConfigDict(extra="forbid")
    discard_runs: Annotated[int, Field(ge=0)] = 0


class BenchmarkSpec(BaseModel):
    """One transcription benchmark — fixture + scoring metric + metadata."""

    model_config = ConfigDict(extra="forbid")
    benchmark_id: Annotated[str, Field(min_length=1)]
    suite_version: Annotated[str, Field(pattern=r"^\d+\.\d+\.\d+(-[\w.]+)?$")]
    description: str = ""
    modality: Literal["voice"] = "voice"
    kind: Literal["transcription"] = "transcription"
    dataset: DatasetConfig
    slo_template: str = "voice.transcription.standard"
    warmup: WarmupConfig = Field(default_factory=WarmupConfig)
    scoring: Literal["wer", "cer", "exact_match"] = "wer"


class RunContext(BaseModel):
    """Per-invocation context (where to send requests, where to write results).

    Mirrors the llm-quality plugin shape so cross-plugin tooling can reuse
    the same context object.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
    model_id: Annotated[str, Field(min_length=1)]
    model_revision: Annotated[str, Field(min_length=7, max_length=40)] = "unknown00"
    engine_kind: EngineKind
    engine_version: str = ""
    base_url: str = ""
    api_key: str = ""
    quantization_format: str = ""
    hardware_class: str = ""
    output_dir: Path
    extra: dict[str, str | int | float | bool] = Field(default_factory=dict)
