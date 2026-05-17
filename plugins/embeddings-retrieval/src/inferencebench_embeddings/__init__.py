"""InferenceBench embeddings-retrieval plugin."""

from inferencebench_embeddings.plugin import EmbeddingsRetrievalPlugin
from inferencebench_embeddings.schemas import BenchmarkSpec, EngineKind, RunContext

__all__ = ["BenchmarkSpec", "EmbeddingsRetrievalPlugin", "EngineKind", "RunContext"]
