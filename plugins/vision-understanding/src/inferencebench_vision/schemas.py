"""Pydantic schemas for vision-understanding benchmark specs + run context."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class EngineKind(StrEnum):
    """Engines this plugin can drive.

    Modern vision-language model serving is dominated by OpenAI-compatible
    endpoints (vLLM, SGLang) and provider-hosted multimodal APIs (OpenAI,
    Anthropic). All four accept the same ``messages[].content`` list-of-parts
    request shape, so they share one plugin code path.
    """

    VLLM = "vllm"
    SGLANG = "sglang"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class DatasetConfig(BaseModel):
    """Dataset under evaluation.

    For the vision plugin the dataset is a small bundled JSONL fixture with
    one ``{"image_path", "question", "answer", "task"}`` object per line.
    Image paths are resolved relative to the plugin's ``datasets/`` directory.
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

    Vision scoring is per-question and order-independent, so the default is
    zero discarded runs. Surfaced as a knob for parity with the other plugins.
    """

    model_config = ConfigDict(extra="forbid")
    discard_runs: Annotated[int, Field(ge=0)] = 0


class BenchmarkSpec(BaseModel):
    """One vision-understanding benchmark — fixture + scoring strategy + metadata."""

    model_config = ConfigDict(extra="forbid")
    benchmark_id: Annotated[str, Field(min_length=1)]
    suite_version: Annotated[str, Field(pattern=r"^\d+\.\d+\.\d+(-[\w.]+)?$")]
    description: str = ""
    modality: Literal["vision"] = "vision"
    kind: Literal["understanding"] = "understanding"
    dataset: DatasetConfig
    slo_template: str = "vision.standard"
    warmup: WarmupConfig = Field(default_factory=WarmupConfig)
    scoring: Literal["exact_match", "substring_match", "judge_llm"] = "substring_match"
    judge_model: str | None = Field(
        default=None,
        description=(
            "Model id used as the LLM judge when scoring == 'judge_llm'. "
            "Falls back to RunContext.extra['judge_model'] or "
            "'openai/gpt-4o-mini' when unset."
        ),
    )


class RunContext(BaseModel):
    """Per-invocation context (where to send requests, where to write results).

    Mirrors the llm-quality plugin so cross-plugin tooling can reuse the same
    context object shape.
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
