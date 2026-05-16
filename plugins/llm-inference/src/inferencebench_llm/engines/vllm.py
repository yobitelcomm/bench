"""vLLM engine adapter.

vLLM exposes an OpenAI-compatible HTTP server. We treat it as such: probe its
``/v1/models`` endpoint for liveness + version, then drive requests via
LiteLLM with ``openai/`` prefix.

The version comes from ``/v1/models`` if available, else from a fallback HTTP
header. If the server isn't reachable we raise :class:`EngineUnavailableError`
with a specific diagnostic.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from inferencebench.harness import ModelClient
from inferencebench_llm.engines.base import Engine, EngineUnavailableError
from inferencebench_llm.schemas import RunContext


class VLLMEngine(Engine):
    """vLLM OpenAI-compatible server adapter."""

    name = "vllm"

    def probe(self, context: RunContext) -> str:
        """Hit ``/v1/models`` to confirm the server is alive + read the version.

        We accept a ``base_url`` of either ``http://host:port`` or
        ``http://host:port/v1`` — strip the trailing ``/v1`` for the models call.
        """
        if not context.base_url:
            msg = (
                "vLLM engine requires --endpoint / context.base_url "
                "(e.g. http://localhost:8000/v1)."
            )
            raise EngineUnavailableError(msg)

        base = context.base_url.rstrip("/")
        if base.endswith("/v1"):
            base = base[: -len("/v1")]

        models_url = f"{base}/v1/models"
        version = "unknown"
        try:
            req = urllib.request.Request(models_url)
            if context.api_key:
                req.add_header("Authorization", f"Bearer {context.api_key}")
            with urllib.request.urlopen(req, timeout=5) as resp:
                body = resp.read().decode("utf-8", errors="replace")
        except urllib.error.URLError as exc:
            msg = (
                f"vLLM endpoint not reachable at {models_url}: {exc.reason}. "
                "Is the server running? `vllm serve <model> --port 8000`"
            )
            raise EngineUnavailableError(msg) from exc
        except (TimeoutError, OSError) as exc:
            msg = f"vLLM endpoint timed out at {models_url}: {exc}"
            raise EngineUnavailableError(msg) from exc

        try:
            payload = json.loads(body)
        except json.JSONDecodeError as exc:
            msg = f"vLLM /v1/models returned non-JSON: {body[:120]}"
            raise EngineUnavailableError(msg) from exc

        # vLLM's /v1/models response does not include the server version
        # (``owned_by`` is the literal string "vllm" — the org, not the version).
        # We accept an explicit ``version`` key if a future release adds one.
        data = payload.get("data") or []
        if data and isinstance(data, list):
            first = data[0]
            v = first.get("version")
            if isinstance(v, str) and v:
                return v

        # Fall back to /version which vLLM ships with a JSON {"version": "x.y.z"}.
        try:
            req = urllib.request.Request(f"{base}/version")
            if context.api_key:
                req.add_header("Authorization", f"Bearer {context.api_key}")
            with urllib.request.urlopen(req, timeout=3) as resp:
                v_body = resp.read().decode("utf-8", errors="replace")
            v_payload = json.loads(v_body)
            v_val = v_payload.get("version")
            if isinstance(v_val, str) and v_val:
                return v_val
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
            pass

        return version

    def build_client(self, context: RunContext) -> ModelClient:
        """Return a ModelClient that talks to this vLLM server.

        LiteLLM treats any OpenAI-compatible server as ``openai/<model>``;
        the actual routing happens via ``base_url``. We strip any user-supplied
        ``openai/`` prefix first so we never send ``openai/openai/<model>``.
        """
        base = context.base_url.rstrip("/")
        if not base.endswith("/v1"):
            base = base + "/v1"

        model_id = context.model_id
        if model_id.startswith("openai/"):
            model_id = model_id[len("openai/") :]

        return ModelClient(
            model=f"openai/{model_id}",
            api_key=context.api_key or "EMPTY",  # vLLM ignores but LiteLLM requires non-empty
            base_url=base,
        )
