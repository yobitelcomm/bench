"""Pydantic schemas for benchmark specs + run context."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class EngineKind(StrEnum):
    """Inference engines this plugin can drive. Phase 1: vLLM only."""

    VLLM = "vllm"
    SGLANG = "sglang"  # Phase 2+
    TRTLLM = "trtllm"  # Phase 2+
    LLAMACPP = "llamacpp"  # Phase 2+
    MLX = "mlx"  # Phase 2+


class DatasetSamplingConfig(BaseModel):
    """How to sample the dataset before driving requests."""

    model_config = ConfigDict(extra="forbid")
    n: Annotated[int, Field(ge=1, description="Number of prompts to sample.")]
    seed: int = 42


class DatasetConfig(BaseModel):
    """Dataset under evaluation."""

    model_config = ConfigDict(extra="forbid")
    id: Annotated[str, Field(min_length=1)]
    uri: Annotated[str, Field(min_length=1, description="hf://, file://, https:// URI")]
    hash: Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")]
    sampling: DatasetSamplingConfig


class DriverConfig(BaseModel):
    """Driver parameters: open-loop or closed-loop, rate, duration."""

    model_config = ConfigDict(extra="forbid")
    type: Literal["open_loop", "closed_loop"]
    arrival: Literal["poisson"] = "poisson"
    rps: list[float] = Field(
        default_factory=list, description="For open-loop (one entry per RPS to sweep)."
    )
    concurrency: list[int] = Field(default_factory=list, description="For closed-loop.")
    duration_s: Annotated[int, Field(ge=1)]


class WarmupConfig(BaseModel):
    """Warmup + convergence-gate parameters."""

    model_config = ConfigDict(extra="forbid")
    discard_runs: Annotated[int, Field(ge=0)] = 3
    convergence_cov_threshold: float = 0.05
    convergence_window: Annotated[int, Field(ge=2)] = 30


class BenchmarkSpec(BaseModel):
    """One benchmark — what to run, against what dataset, with which metrics."""

    model_config = ConfigDict(extra="forbid")
    benchmark_id: Annotated[str, Field(min_length=1)]
    suite_version: Annotated[str, Field(pattern=r"^\d+\.\d+\.\d+(-[\w.]+)?$")]
    description: str = ""
    modality: Literal["llm"] = "llm"
    kind: Literal["perf", "quality", "both"] = "perf"
    dataset: DatasetConfig
    driver: DriverConfig
    slo_template: str = "llm.standard"
    metrics: list[str] = Field(default_factory=list)
    warmup: WarmupConfig = Field(default_factory=WarmupConfig)


class RunContext(BaseModel):
    """Per-invocation context (where to send requests, where to write results)."""

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
