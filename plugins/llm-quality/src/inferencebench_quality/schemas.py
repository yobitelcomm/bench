"""Pydantic schemas for llm-quality benchmark specs + run context."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class EngineKind(StrEnum):
    """Engines this plugin can drive.

    Quality scoring is dominated by per-prompt API calls, so a provider-hosted
    OpenAI-style endpoint is the most useful third option alongside the
    self-hosted vLLM / SGLang servers exercised by the perf plugin.
    """

    VLLM = "vllm"
    SGLANG = "sglang"
    OPENAI = "openai"


class DatasetConfig(BaseModel):
    """Dataset under evaluation.

    For the quality plugin the dataset is a small bundled JSONL fixture with
    one ``{"question", "answer", "category"}`` object per line. Multi-turn
    benchmarks (``persona_consistency`` / ``judge_llm_persona``) use a
    different per-row shape: ``{"case_id", "system_prompt", "markers",
    "turns": [{"question", ...}]}``.
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

    Quality scoring is per-question and order-independent, so the default is
    zero discarded runs. Surfaced as a knob so future revisions can warm up
    a JIT-compiled judge model if needed.
    """

    model_config = ConfigDict(extra="forbid")
    discard_runs: Annotated[int, Field(ge=0)] = 0


class BenchmarkSpec(BaseModel):
    """One quality benchmark — fixture + scoring strategy + metadata."""

    model_config = ConfigDict(extra="forbid")
    benchmark_id: Annotated[str, Field(min_length=1)]
    suite_version: Annotated[str, Field(pattern=r"^\d+\.\d+\.\d+(-[\w.]+)?$")]
    description: str = ""
    modality: Literal["llm"] = "llm"
    kind: Literal["quality"] = "quality"
    dataset: DatasetConfig
    slo_template: str = "llm.quality.standard"
    warmup: WarmupConfig = Field(default_factory=WarmupConfig)
    scoring: Literal[
        "exact_match",
        "substring_match",
        "f1_token",
        "judge_llm",
        "persona_consistency",
        "judge_llm_persona",
    ] = "substring_match"
    judge_model: str | None = Field(
        default=None,
        description=(
            "Model id used as the LLM judge when scoring == 'judge_llm' or "
            "'judge_llm_persona'. Falls back to RunContext.extra['judge_model'] "
            "or 'openai/gpt-4o-mini' when unset."
        ),
    )
    multi_turn: bool = Field(
        default=False,
        description=(
            "When True, the plugin's run() switches to the multi-turn path. "
            "Fixture rows then carry a (system_prompt, turns, markers) shape "
            "and scoring must be one of the persona scorers."
        ),
    )


class RunContext(BaseModel):
    """Per-invocation context (where to send requests, where to write results).

    Mirrors the llm-inference plugin so cross-plugin tooling can reuse the
    same context object shape.
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
