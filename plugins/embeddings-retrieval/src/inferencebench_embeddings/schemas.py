"""Pydantic schemas for embeddings-retrieval benchmark specs + run context."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class EngineKind(StrEnum):
    """Engines this plugin can drive.

    Production-grade paths for embeddings: HuggingFace's Text Embeddings
    Inference (TEI) for self-hosted, plus the two big provider-hosted
    options (OpenAI text-embedding-3, Cohere embed-english).
    """

    TEI = "tei"
    OPENAI = "openai"
    COHERE = "cohere"


class DatasetConfig(BaseModel):
    """Dataset under evaluation.

    Fixture is a JSONL of query records — each with the relevant doc-id set
    and a pointer to the corpus JSONL. Corpus is a sibling JSONL containing
    ``{"doc_id", "text"}`` rows.
    """

    model_config = ConfigDict(extra="forbid")
    id: Annotated[str, Field(min_length=1)]
    path: Annotated[
        str,
        Field(
            min_length=1,
            description=(
                "Path to the queries JSONL relative to the plugin's datasets/ directory."
            ),
        ),
    ]
    corpus_path: Annotated[
        str,
        Field(
            min_length=1,
            description=(
                "Path to the corpus JSONL relative to the plugin's datasets/ directory."
            ),
        ),
    ]


class WarmupConfig(BaseModel):
    """Warmup parameters.

    Retrieval runs are per-query and order-independent; default is zero
    discarded runs. Surfaced for future JIT-warmup of embedding servers.
    """

    model_config = ConfigDict(extra="forbid")
    discard_runs: Annotated[int, Field(ge=0)] = 0


class BenchmarkSpec(BaseModel):
    """One retrieval benchmark — fixture + metric + metadata."""

    model_config = ConfigDict(extra="forbid")
    benchmark_id: Annotated[str, Field(min_length=1)]
    suite_version: Annotated[str, Field(pattern=r"^\d+\.\d+\.\d+(-[\w.]+)?$")]
    description: str = ""
    modality: Literal["embeddings"] = "embeddings"
    kind: Literal["retrieval"] = "retrieval"
    dataset: DatasetConfig
    slo_template: str = "embeddings.retrieval.standard"
    warmup: WarmupConfig = Field(default_factory=WarmupConfig)
    metric: Literal["recall_at_5", "mrr_at_10", "ndcg_at_10"] = "recall_at_5"


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
