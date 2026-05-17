"""llama.cpp engine adapter.

``llama-server`` (shipped with llama.cpp) exposes an OpenAI-compatible HTTP
server, so the bulk of this adapter mirrors :mod:`inferencebench_llm.engines.vllm`.
The only meaningful difference is the version probe: llama.cpp publishes its
launch metadata at ``/props`` (a JSON document with ``system_info``,
``model_path``, ``n_ctx`` and friends). There's no clean version string —
``system_info`` is a build-time CPU-feature dump (``AVX2 1 | AVX_VNNI 0 | ...``)
which we surface verbatim as the best liveness signal llama.cpp gives us
without a git-hash injection.

If ``/props`` is unavailable we fall back to ``/v1/models`` (which every
OpenAI-compatible server implements) for a pure liveness check and accept
an "unknown" version.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from inferencebench.harness import ModelClient
from inferencebench_llm.engines.base import Engine, EngineUnavailableError
from inferencebench_llm.schemas import RunContext


class LlamaCppEngine(Engine):
    """llama.cpp (``llama-server``) OpenAI-compatible server adapter."""

    name = "llamacpp"

    def probe(self, context: RunContext) -> str:
        """Hit ``/props`` to confirm llama-server is alive + read a liveness signal.

        ``base_url`` may be ``http://host:port`` or ``http://host:port/v1`` —
        strip a trailing ``/v1`` since ``/props`` is rooted, not under ``/v1``.
        """
        if not context.base_url:
            msg = (
                "llama.cpp engine requires --endpoint / context.base_url "
                "(e.g. http://localhost:8000/v1)."
            )
            raise EngineUnavailableError(msg)

        base = context.base_url.rstrip("/")
        if base.endswith("/v1"):
            base = base[: -len("/v1")]

        props_url = f"{base}/props"
        try:
            req = urllib.request.Request(props_url)
            if context.api_key:
                req.add_header("Authorization", f"Bearer {context.api_key}")
            with urllib.request.urlopen(req, timeout=5) as resp:
                body = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            # 404 / 405 — older llama-server or unrelated OpenAI-compat server.
            # Fall through to the /v1/models liveness check.
            if exc.code in (404, 405):
                return self._probe_via_models(base, context)
            msg = (
                f"llama.cpp endpoint returned HTTP {exc.code} at {props_url}: "
                f"{exc.reason}. Is llama-server running? "
                "`./llama-server -m <model.gguf> --port 8000`"
            )
            raise EngineUnavailableError(msg) from exc
        except urllib.error.URLError as exc:
            msg = (
                f"llama.cpp endpoint not reachable at {props_url}: {exc.reason}. "
                "Is llama-server running? `./llama-server -m <model.gguf> --port 8000`"
            )
            raise EngineUnavailableError(msg) from exc
        except (TimeoutError, OSError) as exc:
            msg = f"llama.cpp endpoint timed out at {props_url}: {exc}"
            raise EngineUnavailableError(msg) from exc

        try:
            payload = json.loads(body)
        except json.JSONDecodeError as exc:
            msg = f"llama.cpp /props returned non-JSON: {body[:120]}"
            raise EngineUnavailableError(msg) from exc

        system_info = payload.get("system_info")
        model_path = payload.get("model_path")
        if isinstance(system_info, str) and system_info and (
            "llama.cpp" in system_info or isinstance(model_path, str)
        ):
            # First 60 chars of system_info is the best liveness signal
            # llama.cpp provides without a git hash. It's not a version per
            # se but it pins the CPU-feature/build context for the envelope.
            return system_info[:60]
        return "unknown"

    def _probe_via_models(self, base: str, context: RunContext) -> str:
        """Liveness fallback using ``/v1/models``.

        Returns ``"unknown"`` on success — ``/v1/models`` doesn't expose the
        server version. Raises :class:`EngineUnavailableError` on failure.
        """
        models_url = f"{base}/v1/models"
        try:
            req = urllib.request.Request(models_url)
            if context.api_key:
                req.add_header("Authorization", f"Bearer {context.api_key}")
            with urllib.request.urlopen(req, timeout=5) as resp:
                resp.read()
        except urllib.error.URLError as exc:
            msg = (
                f"llama.cpp endpoint not reachable at {models_url}: {exc.reason}. "
                "Is llama-server running? `./llama-server -m <model.gguf> --port 8000`"
            )
            raise EngineUnavailableError(msg) from exc
        except (TimeoutError, OSError) as exc:
            msg = f"llama.cpp endpoint timed out at {models_url}: {exc}"
            raise EngineUnavailableError(msg) from exc
        return "unknown"

    def build_client(self, context: RunContext) -> ModelClient:
        """Return a ModelClient that talks to this llama-server.

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

        # llama-server ignores api_key but LiteLLM requires non-empty.
        return ModelClient(
            model=f"openai/{model_id}",
            api_key=context.api_key or "EMPTY",
            base_url=base,
        )
