"""TensorRT-LLM engine adapter.

NVIDIA TensorRT-LLM ships ``trtllm-serve``, an OpenAI-compatible HTTP server
that listens on port ``8000`` by default. The bulk of this adapter mirrors
:mod:`inferencebench_llm.engines.vllm` and :mod:`inferencebench_llm.engines.sglang`.

The version probe is unusual: TRT-LLM does not expose a JSON
``/version`` (or equivalent) endpoint. The closest thing is the ``Server``
response header on ``/v1/models`` — recent builds emit
``Server: trtllm-{version}`` — but the header isn't always present.

Probe order:

1. ``GET <base>/health/load`` — TRT-LLM's metadata/liveness endpoint. A 200
   response confirms the server is alive; we then try to parse the ``Server``
   header (``trtllm-0.13.0`` -> ``0.13.0``).
2. If ``/health/load`` returns 404/405 (older builds), or the ``Server``
   header is missing/unparseable, fall back to ``GET <base>/v1/models`` for
   a pure liveness check, returning ``"unknown"`` for the version.
"""

from __future__ import annotations

import urllib.error
import urllib.request

from inferencebench.harness import ModelClient
from inferencebench_llm.engines.base import Engine, EngineUnavailableError
from inferencebench_llm.schemas import RunContext

_SERVER_HEADER_PREFIX = "trtllm-"


def _parse_server_header(value: str | None) -> str | None:
    """Return the version embedded in a ``Server: trtllm-<ver>`` header.

    Returns ``None`` if ``value`` is missing, empty, or doesn't carry the
    ``trtllm-`` token (case-insensitive). The token may appear after other
    server identifiers (rare), so we scan whitespace-separated parts.
    """
    if not value:
        return None
    for part in value.split():
        lowered = part.lower()
        if lowered.startswith(_SERVER_HEADER_PREFIX):
            ver = part[len(_SERVER_HEADER_PREFIX) :]
            if ver:
                return ver
    return None


class TRTLLMEngine(Engine):
    """TensorRT-LLM (``trtllm-serve``) OpenAI-compatible server adapter."""

    name = "trtllm"

    def probe(self, context: RunContext) -> str:
        """Hit ``/health/load`` to confirm ``trtllm-serve`` is alive + read the version.

        ``base_url`` may be ``http://host:port`` or ``http://host:port/v1`` —
        strip a trailing ``/v1`` since ``/health/load`` is rooted, not under
        ``/v1``.
        """
        if not context.base_url:
            msg = (
                "TRT-LLM engine requires --endpoint / context.base_url "
                "(e.g. http://localhost:8000/v1)."
            )
            raise EngineUnavailableError(msg)

        base = context.base_url.rstrip("/")
        if base.endswith("/v1"):
            base = base[: -len("/v1")]

        load_url = f"{base}/health/load"
        try:
            req = urllib.request.Request(load_url)
            req.add_header("User-Agent", "inferencebench-trtllm")
            if context.api_key:
                req.add_header("Authorization", f"Bearer {context.api_key}")
            with urllib.request.urlopen(req, timeout=5) as resp:
                resp.read()
                server_header = resp.headers.get("Server")
        except urllib.error.HTTPError as exc:
            # 404 / 405 — older trtllm-serve or unrelated OpenAI-compat server.
            # Fall through to the /v1/models liveness check.
            if exc.code in (404, 405):
                return self._probe_via_models(base, context)
            msg = (
                f"TRT-LLM endpoint returned HTTP {exc.code} at {load_url}: "
                f"{exc.reason}. Is TRT-LLM running? `trtllm-serve <engine_dir>`"
            )
            raise EngineUnavailableError(msg) from exc
        except urllib.error.URLError as exc:
            msg = (
                f"TRT-LLM endpoint not reachable at {load_url}: {exc.reason}. "
                "Is TRT-LLM running? `trtllm-serve <engine_dir>`"
            )
            raise EngineUnavailableError(msg) from exc
        except (TimeoutError, OSError) as exc:
            msg = f"TRT-LLM endpoint timed out at {load_url}: {exc}"
            raise EngineUnavailableError(msg) from exc

        parsed = _parse_server_header(server_header)
        if parsed:
            return parsed
        # /health/load is alive but didn't carry a usable Server header.
        # Try /v1/models one more time — some builds set the header there
        # but not on /health/load.
        return self._probe_via_models(base, context)

    def _probe_via_models(self, base: str, context: RunContext) -> str:
        """Liveness fallback using ``/v1/models``.

        Returns the version parsed from the ``Server`` header if present,
        otherwise ``"unknown"``. Raises :class:`EngineUnavailableError` on
        connection failure.
        """
        models_url = f"{base}/v1/models"
        try:
            req = urllib.request.Request(models_url)
            req.add_header("User-Agent", "inferencebench-trtllm")
            if context.api_key:
                req.add_header("Authorization", f"Bearer {context.api_key}")
            with urllib.request.urlopen(req, timeout=5) as resp:
                resp.read()
                server_header = resp.headers.get("Server")
        except urllib.error.URLError as exc:
            msg = (
                f"TRT-LLM endpoint not reachable at {models_url}: {exc.reason}. "
                "Is TRT-LLM running? `trtllm-serve <engine_dir>`"
            )
            raise EngineUnavailableError(msg) from exc
        except (TimeoutError, OSError) as exc:
            msg = f"TRT-LLM endpoint timed out at {models_url}: {exc}"
            raise EngineUnavailableError(msg) from exc

        parsed = _parse_server_header(server_header)
        if parsed:
            return parsed
        return "unknown"

    def build_client(self, context: RunContext) -> ModelClient:
        """Return a ModelClient that talks to this ``trtllm-serve`` instance.

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

        # trtllm-serve ignores api_key but LiteLLM requires non-empty.
        return ModelClient(
            model=f"openai/{model_id}",
            api_key=context.api_key or "EMPTY",
            base_url=base,
        )
