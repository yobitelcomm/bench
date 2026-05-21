"""SGLang engine adapter.

SGLang exposes an OpenAI-compatible HTTP server, so the bulk of this adapter
mirrors :mod:`inferencebench_llm.engines.vllm`. The only meaningful difference
is the version probe: SGLang publishes its launch metadata at
``/get_server_info`` (a JSON document with a top-level ``version`` key and a
``server_args`` block) rather than a ``/version`` endpoint.

If ``/get_server_info`` is unavailable we fall back to ``/v1/models`` (which
every OpenAI-compatible server implements) for a pure liveness check and
accept an "unknown" version.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from inferencebench.harness import ModelClient
from inferencebench_llm.engines.base import Engine, EngineUnavailableError
from inferencebench_llm.schemas import RunContext


class SGLangEngine(Engine):
    """SGLang OpenAI-compatible server adapter."""

    name = "sglang"

    def probe(self, context: RunContext) -> str:
        """Hit ``/get_server_info`` to confirm SGLang is alive + read the version.

        ``base_url`` may be ``http://host:port`` or ``http://host:port/v1`` —
        strip a trailing ``/v1`` since ``/get_server_info`` is rooted, not
        under ``/v1``.
        """
        if not context.base_url:
            msg = (
                "SGLang engine requires --endpoint / context.base_url "
                "(e.g. http://localhost:30000/v1)."
            )
            raise EngineUnavailableError(msg)

        base = context.base_url.rstrip("/")
        if base.endswith("/v1"):
            base = base[: -len("/v1")]

        info_url = f"{base}/get_server_info"
        try:
            req = urllib.request.Request(info_url)
            if context.api_key:
                req.add_header("Authorization", f"Bearer {context.api_key}")
            with urllib.request.urlopen(req, timeout=5) as resp:
                body = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            # 404 / 405 — older SGLang or unrelated OpenAI-compat server.
            # Fall through to the /v1/models liveness check.
            if exc.code in (404, 405):
                return self._probe_via_models(base, context)
            msg = (
                f"SGLang endpoint returned HTTP {exc.code} at {info_url}: "
                f"{exc.reason}. Is SGLang running?"
            )
            raise EngineUnavailableError(msg) from exc
        except urllib.error.URLError as exc:
            msg = (
                f"SGLang endpoint not reachable at {info_url}: {exc.reason}. "
                "Is SGLang running? `python -m sglang.launch_server --model-path "
                "<model> --port 30000`"
            )
            raise EngineUnavailableError(msg) from exc
        except (TimeoutError, OSError) as exc:
            msg = f"SGLang endpoint timed out at {info_url}: {exc}"
            raise EngineUnavailableError(msg) from exc

        try:
            payload = json.loads(body)
        except json.JSONDecodeError as exc:
            msg = f"SGLang /get_server_info returned non-JSON: {body[:120]}"
            raise EngineUnavailableError(msg) from exc

        v = payload.get("version")
        if isinstance(v, str) and v:
            return v
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
            msg = f"SGLang endpoint not reachable at {models_url}: {exc.reason}. Is SGLang running?"
            raise EngineUnavailableError(msg) from exc
        except (TimeoutError, OSError) as exc:
            msg = f"SGLang endpoint timed out at {models_url}: {exc}"
            raise EngineUnavailableError(msg) from exc
        return "unknown"

    def build_client(self, context: RunContext) -> ModelClient:
        """Return a ModelClient that talks to this SGLang server.

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

        return ModelClient(
            model=f"openai/{model_id}",
            api_key=context.api_key or "EMPTY",  # SGLang ignores but LiteLLM requires non-empty
            base_url=base,
        )
