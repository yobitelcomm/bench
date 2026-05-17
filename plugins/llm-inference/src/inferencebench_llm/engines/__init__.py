"""Engine adapters for LLM inference benchmarking.

Each engine is a thin shim that knows how to:
- Verify the engine is reachable / installed
- Read the engine's version string for envelope provenance
- Construct a :class:`ModelClient` pointed at it

Ships :class:`VLLMEngine`, :class:`SGLangEngine`, :class:`LlamaCppEngine`,
:class:`TRTLLMEngine`, and :class:`MLXEngine` — the full five-engine matrix.
"""

from inferencebench_llm.engines.base import Engine, EngineUnavailableError
from inferencebench_llm.engines.llamacpp import LlamaCppEngine
from inferencebench_llm.engines.mlx import MLXEngine
from inferencebench_llm.engines.sglang import SGLangEngine
from inferencebench_llm.engines.trtllm import TRTLLMEngine
from inferencebench_llm.engines.vllm import VLLMEngine

__all__ = [
    "Engine",
    "EngineUnavailableError",
    "LlamaCppEngine",
    "MLXEngine",
    "SGLangEngine",
    "TRTLLMEngine",
    "VLLMEngine",
]
