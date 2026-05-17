"""Pydantic schemas for llm-mt benchmark specs + run context."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class EngineKind(StrEnum):
    """Engines this plugin can drive.

    Machine translation is dominated by per-prompt API calls, so the four
    most useful endpoints are self-hosted OpenAI-compatible servers (vLLM,
    SGLang), provider-hosted OpenAI, and Cohere — whose Aya / Command
    models are popular MT picks.
    """

    VLLM = "vllm"
    SGLANG = "sglang"
    OPENAI = "openai"
    COHERE = "cohere"


class DatasetConfig(BaseModel):
    """Dataset under evaluation.

    For the MT plugin the dataset is a small bundled JSONL fixture with one
    ``{"source", "reference", "domain"}`` object per line.
    """

    model_config = ConfigDict(extra="forbid")
    id: Annotated[str, Field(min_length=1)]
    path: Annotated[
        str,
        Field(
            min_length=1,
            description=(
                "Path to the fixture JSONL relative to the plugin's datasets/ directory."
            ),
        ),
    ]


class WarmupConfig(BaseModel):
    """Warmup parameters.

    MT scoring is per-sentence and order-independent, so the default is
    zero discarded runs. Surfaced as a knob so future revisions can warm
    up server-side weights if needed.
    """

    model_config = ConfigDict(extra="forbid")
    discard_runs: Annotated[int, Field(ge=0)] = 0


class BenchmarkSpec(BaseModel):
    """One MT benchmark — fixture + scoring strategy + language pair + metadata."""

    model_config = ConfigDict(extra="forbid")
    benchmark_id: Annotated[str, Field(min_length=1)]
    suite_version: Annotated[str, Field(pattern=r"^\d+\.\d+\.\d+(-[\w.]+)?$")]
    description: str = ""
    modality: Literal["llm"] = "llm"
    kind: Literal["translation"] = "translation"
    dataset: DatasetConfig
    slo_template: str = "llm.mt.standard"
    warmup: WarmupConfig = Field(default_factory=WarmupConfig)
    scoring: Literal["chrf", "bleu_token", "exact_match"] = "chrf"
    source_lang: Annotated[str, Field(min_length=2, max_length=8)]
    target_lang: Annotated[str, Field(min_length=2, max_length=8)]


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
