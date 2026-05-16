"""Engine adapter ABC."""

from __future__ import annotations

from abc import ABC, abstractmethod

from inferencebench.harness import ModelClient
from inferencebench_llm.schemas import RunContext


class EngineUnavailableError(Exception):
    """Raised when the engine isn't reachable / installed."""


class Engine(ABC):
    """Adapter for a specific inference engine (vLLM, SGLang, TRT-LLM, ...)."""

    name: str  # populated by subclasses

    @abstractmethod
    def probe(self, context: RunContext) -> str:
        """Verify the engine is reachable and return its version string.

        Raises :class:`EngineUnavailableError` with a diagnostic message
        if the engine cannot be reached. The version string goes into the
        envelope's ``engine.version`` field.
        """

    @abstractmethod
    def build_client(self, context: RunContext) -> ModelClient:
        """Construct a :class:`ModelClient` pointed at this engine."""
