"""InferenceBench embeddings-retrieval plugin."""

from inferencebench_embeddings.plugin import (
    EXPECTED_METRICS,
    EmbeddingsRetrievalPlugin,
)
from inferencebench_embeddings.schemas import BenchmarkSpec, EngineKind, RunContext

__all__ = [
    "EXPECTED_METRICS",
    "BenchmarkSpec",
    "EmbeddingsRetrievalPlugin",
    "EngineKind",
    "RunContext",
]
