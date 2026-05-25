"""MLX engine adapter.

Apple's `mlx-lm` ships ``mlx_lm.server`` which provides an OpenAI-compatible
HTTP API. The server is the entry point for running MLX-format models (and
GGUFs) on Apple Silicon hardware. From the benchmark's perspective it looks
just like vLLM or SGLang: an OpenAI-compatible endpoint with ``/v1/models``,
``/v1/chat/completions`` and friends.

Probe quirks
------------

``mlx_lm.server`` does **not** expose a ``/version`` or ``/health`` route —
only ``/v1/models``. We therefore hit ``/v1/models`` for a liveness check and
return ``"unknown"`` for the version string. The MLX version isn't exposed
via the HTTP API at all, so there's nothing to parse out of headers or
payloads.

Start the server with::

    python -m mlx_lm.server --model <gguf-or-mlx-path> --port 8000
"""

from __future__ import annotations

import urllib.error
import urllib.request

from inferencebench.harness import ModelClient
from inferencebench_llm.engines.base import Engine, EngineUnavailableError
from inferencebench_llm.schemas import RunContext


class MLXEngine(Engine):
    """``mlx_lm.server`` OpenAI-compatible server adapter."""

    name = "mlx"

    def probe(self, context: RunContext) -> str:
        """Hit ``/v1/models`` to confirm the server is alive.

        ``base_url`` may be ``http://host:port`` or ``http://host:port/v1`` —
        we strip a trailing ``/v1`` so we can reconstruct the canonical
        ``/v1/models`` URL exactly once.

        Returns ``"unknown"`` on a successful 200 because mlx_lm.server does
        not expose the engine version via the HTTP API.
        """
        if not context.base_url:
            msg = (
                "MLX engine requires --endpoint / context.base_url (e.g. http://localhost:8000/v1)."
            )
            raise EngineUnavailableError(msg)

        base = context.base_url.rstrip("/")
        if base.endswith("/v1"):
            base = base[: -len("/v1")]

        models_url = f"{base}/v1/models"
        try:
            req = urllib.request.Request(models_url)
            req.add_header("User-Agent", "inferencebench-mlx")
            if context.api_key:
                req.add_header("Authorization", f"Bearer {context.api_key}")
            with urllib.request.urlopen(req, timeout=5) as resp:
                resp.read()
        except urllib.error.URLError as exc:
            msg = (
                f"MLX endpoint not reachable at {models_url}: {exc.reason}. "
                "Is mlx_lm.server running? "
                "`python -m mlx_lm.server --model <gguf-or-mlx-path> --port 8000`"
            )
            raise EngineUnavailableError(msg) from exc
        except (TimeoutError, OSError) as exc:
            msg = (
                f"MLX endpoint timed out at {models_url}: {exc}. "
                "Is mlx_lm.server running? "
                "`python -m mlx_lm.server --model <gguf-or-mlx-path> --port 8000`"
            )
            raise EngineUnavailableError(msg) from exc

        return "unknown"

    def build_client(self, context: RunContext) -> ModelClient:
        """Return a ModelClient that talks to this ``mlx_lm.server`` instance.

        LiteLLM routes any OpenAI-compatible server as ``openai/<model>``
        with the actual host set via ``base_url``. We strip a user-supplied
        ``openai/`` prefix first so we never emit ``openai/openai/<model>``.
        """
        base = context.base_url.rstrip("/")
        if not base.endswith("/v1"):
            base = base + "/v1"

        model_id = context.model_id
        if model_id.startswith("openai/"):
            model_id = model_id[len("openai/") :]

        # mlx_lm.server ignores api_key but LiteLLM requires non-empty.
        return ModelClient(
            model=f"openai/{model_id}",
            api_key=context.api_key or "EMPTY",
            base_url=base,
        )
