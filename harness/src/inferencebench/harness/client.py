"""Unified model invocation via LiteLLM.

`ModelClient` is the single entry point harness drivers use to call any model
endpoint — OpenAI, Anthropic, Google, vLLM-local, SGLang-local, Together, Groq,
Fireworks, Bedrock, Vertex, etc. LiteLLM normalises the 100+ provider APIs.

This wrapper adds:
- Streaming-aware latency capture (TTFT, total latency)
- Token counting (provider-reported when available, tiktoken fallback)
- Cost extraction from provider response, or computed from a registry (Phase 2+)
- Structured errors, never silent swallowing

Public API::

    from inferencebench.harness.client import ModelClient, CompletionResult

    client = ModelClient(
        model="openai/gpt-4o-mini",
        api_key="...",
    )
    result = client.complete("Hello, world.", max_tokens=50, stream=True)
    print(result.text, result.ttft_ms, result.tokens_out)
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any


class ClientError(Exception):
    """Raised when the underlying provider call fails."""


@dataclass(frozen=True, slots=True)
class CompletionResult:
    """Outcome of one `complete()` call.

    All timings are in milliseconds with perf_counter precision.
    Token counts are best-effort: provider-reported when available, else
    tiktoken-estimated; ``token_source`` records which.
    """

    text: str
    tokens_in: int
    tokens_out: int
    ttft_ms: float
    total_ms: float
    tpot_ms: float
    cost_usd: float
    finish_reason: str
    token_source: str  # "provider" | "tiktoken" | "approx-words"
    raw: dict[str, Any] = field(default_factory=dict)


class ModelClient:
    """Provider-neutral model invocation wrapper.

    Args:
        model: Full LiteLLM model id, e.g. ``"openai/gpt-4o-mini"``,
            ``"anthropic/claude-opus-4-7"``, ``"openai/Llama-4-Maverick"``
            (for a vLLM endpoint via OpenAI-compatible mode).
        api_key: Provider API key. If None, falls back to standard env vars
            (``OPENAI_API_KEY``, ``ANTHROPIC_API_KEY``, etc.).
        base_url: Endpoint override (e.g. ``http://localhost:8000/v1`` for vLLM).
        timeout_s: Per-request timeout. Default 120 s.
        extra_litellm_kwargs: Forwarded verbatim to litellm.completion().
    """

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_s: float = 120.0,
        extra_litellm_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.timeout_s = timeout_s
        self.extra_litellm_kwargs = dict(extra_litellm_kwargs or {})

    # ---------------------------------------------------------------- API #
    def complete(
        self,
        prompt: str,
        *,
        max_tokens: int = 256,
        temperature: float = 0.0,
        stream: bool = True,
        system: str | None = None,
        **extra: Any,
    ) -> CompletionResult:
        """Run a single completion. Returns ``CompletionResult`` with all metrics."""
        try:
            import litellm
        except ImportError as exc:
            msg = "litellm is required. Install with: pip install inferencebench-harness"
            raise ClientError(msg) from exc

        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
            "timeout": self.timeout_s,
            **self.extra_litellm_kwargs,
            **extra,
        }
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.base_url:
            kwargs["api_base"] = self.base_url

        if stream:
            return self._complete_streaming(litellm, kwargs, prompt)
        return self._complete_blocking(litellm, kwargs, prompt)

    # --------------------------------------------------- impl: streaming #
    def _complete_streaming(
        self,
        litellm: Any,
        kwargs: dict[str, Any],
        prompt: str,
    ) -> CompletionResult:
        t0 = time.perf_counter()
        first_chunk_t: float | None = None
        chunks: list[str] = []
        finish_reason = "unknown"
        provider_tokens_in: int | None = None
        provider_tokens_out: int | None = None
        provider_cost: float | None = None
        raw: dict[str, Any] = {}

        try:
            response = litellm.completion(**kwargs)
            for chunk in response:
                if first_chunk_t is None:
                    first_chunk_t = time.perf_counter()
                try:
                    delta = chunk.choices[0].delta.content or ""
                except (AttributeError, IndexError, TypeError):
                    delta = ""
                if delta:
                    chunks.append(delta)
                fr = getattr(chunk.choices[0], "finish_reason", None) if chunk.choices else None
                if fr:
                    finish_reason = fr
                # Usage is typically only present on the LAST chunk
                usage = getattr(chunk, "usage", None)
                if usage:
                    provider_tokens_in = getattr(usage, "prompt_tokens", provider_tokens_in)
                    provider_tokens_out = getattr(usage, "completion_tokens", provider_tokens_out)
                cost = getattr(chunk, "_hidden_params", {}).get("response_cost")
                if cost is not None:
                    provider_cost = float(cost)
        except Exception as exc:
            msg = f"streaming completion failed: {exc}"
            raise ClientError(msg) from exc

        t1 = time.perf_counter()
        text = "".join(chunks)

        ttft_ms = ((first_chunk_t or t1) - t0) * 1000.0
        total_ms = (t1 - t0) * 1000.0
        tokens_in, tokens_out, token_source = self._resolve_tokens(
            prompt, text, provider_tokens_in, provider_tokens_out
        )
        tpot_ms = (total_ms - ttft_ms) / max(tokens_out - 1, 1)
        cost_usd = provider_cost if provider_cost is not None else 0.0

        return CompletionResult(
            text=text,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            ttft_ms=ttft_ms,
            total_ms=total_ms,
            tpot_ms=tpot_ms,
            cost_usd=cost_usd,
            finish_reason=finish_reason,
            token_source=token_source,
            raw=raw,
        )

    # ----------------------------------------------------- impl: blocking #
    def _complete_blocking(
        self,
        litellm: Any,
        kwargs: dict[str, Any],
        prompt: str,
    ) -> CompletionResult:
        t0 = time.perf_counter()
        try:
            response = litellm.completion(**kwargs)
        except Exception as exc:
            msg = f"blocking completion failed: {exc}"
            raise ClientError(msg) from exc
        t1 = time.perf_counter()

        try:
            text = response.choices[0].message.content or ""
            finish_reason = response.choices[0].finish_reason or "unknown"
        except (AttributeError, IndexError, TypeError) as exc:
            msg = f"unexpected response shape: {exc}"
            raise ClientError(msg) from exc

        usage = getattr(response, "usage", None)
        provider_tokens_in = getattr(usage, "prompt_tokens", None) if usage else None
        provider_tokens_out = getattr(usage, "completion_tokens", None) if usage else None
        provider_cost = getattr(response, "_hidden_params", {}).get("response_cost")

        total_ms = (t1 - t0) * 1000.0
        ttft_ms = total_ms  # blocking: TTFT == total
        tokens_in, tokens_out, token_source = self._resolve_tokens(
            prompt, text, provider_tokens_in, provider_tokens_out
        )
        tpot_ms = total_ms / max(tokens_out, 1)
        cost_usd = float(provider_cost) if provider_cost is not None else 0.0

        return CompletionResult(
            text=text,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            ttft_ms=ttft_ms,
            total_ms=total_ms,
            tpot_ms=tpot_ms,
            cost_usd=cost_usd,
            finish_reason=finish_reason,
            token_source=token_source,
        )

    # ----------------------------------------------------- token resolve #
    def _resolve_tokens(
        self,
        prompt: str,
        completion: str,
        provider_in: int | None,
        provider_out: int | None,
    ) -> tuple[int, int, str]:
        """Return (tokens_in, tokens_out, source). Prefer provider counts."""
        if provider_in is not None and provider_out is not None:
            return int(provider_in), int(provider_out), "provider"

        # tiktoken fallback (works for OpenAI-style BPE; rough estimate elsewhere)
        try:
            import tiktoken

            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(prompt)), len(enc.encode(completion)), "tiktoken"
        except (ImportError, Exception):
            pass

        # Last-resort heuristic: words * 1.3
        return (
            int(len(prompt.split()) * 1.3),
            int(len(completion.split()) * 1.3),
            "approx-words",
        )


def detect_endpoint_health(base_url: str, *, timeout_s: float = 5.0) -> bool:
    """Liveness check for an OpenAI-compatible endpoint (vLLM, SGLang, TGI).

    GETs ``{base_url}/health`` (or ``/`` as fallback). Returns True if the
    endpoint responded with a 2xx status.
    """
    try:
        import urllib.error
        import urllib.request

        for path in ("/health", "/"):
            req = urllib.request.Request(base_url.rstrip("/") + path)
            req.add_header("User-Agent", "inferencebench-harness/health-check")
            try:
                with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                    if 200 <= resp.status < 300:
                        return True
            except urllib.error.URLError:
                continue
            except (TimeoutError, OSError):
                continue
    except ImportError:
        return False
    return False


def env_api_key(provider: str) -> str | None:
    """Return the conventional env-var API key for a given provider.

    Convenience helper for harness CLI integration. Returns None if unset.
    """
    var = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "google": "GOOGLE_API_KEY",
        "groq": "GROQ_API_KEY",
        "together": "TOGETHER_API_KEY",
        "fireworks": "FIREWORKS_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
        "mistral": "MISTRAL_API_KEY",
    }.get(provider.lower())
    if not var:
        return None
    return os.environ.get(var)
