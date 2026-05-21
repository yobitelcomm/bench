"""Pydantic schemas for code-generation benchmark specs + run context."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class EngineKind(StrEnum):
    """Engines this plugin can drive.

    Code-generation scoring is dominated by per-prompt API calls (one model
    invocation per fixture row, then local execution of the response). We
    surface the same four engine kinds the rest of the suite uses so the
    plugin slots into existing cross-vendor comparisons unchanged.
    """

    VLLM = "vllm"
    SGLANG = "sglang"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class DatasetConfig(BaseModel):
    """Dataset under evaluation.

    For the code-generation plugin the dataset is a small bundled JSONL
    fixture; each line is one HumanEval-style task with ``task_id``,
    ``prompt``, ``tests``, ``canonical_solution`` and ``entry_point`` keys.
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

    Code-generation runs are per-task and order-independent so the default
    is zero discarded runs. Knob retained for future revisions (warm-up of
    a JIT-compiled model or sandbox cold-start, etc.).
    """

    model_config = ConfigDict(extra="forbid")
    discard_runs: Annotated[int, Field(ge=0)] = 0


class BenchmarkSpec(BaseModel):
    """One code-generation benchmark — fixture + scoring strategy + metadata."""

    model_config = ConfigDict(extra="forbid")
    benchmark_id: Annotated[str, Field(min_length=1)]
    suite_version: Annotated[str, Field(pattern=r"^\d+\.\d+\.\d+(-[\w.]+)?$")]
    description: str = ""
    modality: Literal["code"] = "code"
    kind: Literal["generation"] = "generation"
    dataset: DatasetConfig
    slo_template: str = "code.generation.standard"
    warmup: WarmupConfig = Field(default_factory=WarmupConfig)
    language: Literal["python"] = "python"
    scoring: Literal["pass_at_1", "pass_at_k"] = "pass_at_1"
    k: Annotated[int, Field(ge=1)] = 1
    timeout_s: Annotated[float, Field(gt=0.0)] = 5.0


class RunContext(BaseModel):
    """Per-invocation context (where to send requests, where to write results).

    Mirrors the llm-quality plugin so cross-plugin tooling can reuse the
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
