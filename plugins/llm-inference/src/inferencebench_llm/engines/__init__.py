"""Engine adapters for LLM inference benchmarking.

Each engine is a thin shim that knows how to:
- Verify the engine is reachable / installed
- Read the engine's version string for envelope provenance
- Construct a :class:`ModelClient` pointed at it

Phase 1 ships :class:`VLLMEngine` and :class:`SGLangEngine`. TensorRT-LLM,
llama.cpp, MLX follow in Phase 2.
"""

from inferencebench_llm.engines.base import Engine, EngineUnavailableError
from inferencebench_llm.engines.sglang import SGLangEngine
from inferencebench_llm.engines.vllm import VLLMEngine

__all__ = ["Engine", "EngineUnavailableError", "SGLangEngine", "VLLMEngine"]
