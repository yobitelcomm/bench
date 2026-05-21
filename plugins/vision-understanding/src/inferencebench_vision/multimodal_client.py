"""Multimodal wrapper around LiteLLM for vision-language completions.

Mirrors :class:`inferencebench.harness.ModelClient` for the text-only case,
but the chat-completions ``messages`` payload is constructed with the
list-of-parts content shape that vLLM, SGLang, the OpenAI API and Anthropic
all accept for image-bearing requests:

    [{"role": "user", "content": [
        {"type": "text", "text": "<question>"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
    ]}]

The image is read from disk, base64-encoded, and wrapped in a data URL with
the MIME type inferred from the file extension. Streaming is disabled because
some endpoints don't yet stream multimodal responses cleanly — ``total_ms``
doubles as TTFT for the non-streaming path.
"""

from __future__ import annotations

import base64
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from inferencebench.harness import CompletionResult
from inferencebench.harness.client import ClientError

if TYPE_CHECKING:
    from inferencebench.harness import ModelClient


# Conservative extension -> MIME mapping. Anything outside this set falls back
# to ``image/png`` because OpenAI/Anthropic both happily accept that for
# arbitrary raster bytes — the data URL declaration is advisory.
_EXT_MIME: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
}


def _detect_mime(image_path: Path) -> str:
    """Return the MIME type for ``image_path`` based on its extension."""
    return _EXT_MIME.get(image_path.suffix.lower(), "image/png")


def _encode_data_url(image_path: Path) -> str:
    """Read ``image_path`` and return a ``data:<mime>;base64,...`` URL."""
    mime = _detect_mime(image_path)
    raw = image_path.read_bytes()
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{b64}"


def build_multimodal_messages(image_path: Path, question: str) -> list[dict[str, Any]]:
    """Build the OpenAI-style multimodal ``messages`` payload.

    Exposed as a free function (rather than tucked inside the client) so the
    plugin's unit tests can build the exact payload that would have been sent
    without having to instantiate a transport.
    """
    data_url = _encode_data_url(image_path)
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        }
    ]


class MultimodalClient:
    """Thin wrapper that issues a multimodal chat completion via LiteLLM.

    Reuses the same ``model`` / ``api_key`` / ``base_url`` resolution that
    :class:`inferencebench.harness.ModelClient` does, but the actual call is
    blocking (``stream=False``) and the ``messages`` payload is the
    list-of-parts shape that ships images inline as base64 data URLs.
    """

    def __init__(self, transport: ModelClient) -> None:
        self._transport = transport

    @property
    def transport(self) -> ModelClient:
        """The underlying text-only :class:`ModelClient` used for routing config."""
        return self._transport

    def complete_multimodal(
        self,
        image_path: Path,
        question: str,
        *,
        max_tokens: int = 256,
    ) -> CompletionResult:
        """Send one (image, question) pair, return a :class:`CompletionResult`.

        Latency is measured around the blocking ``litellm.completion`` call.
        Because the path is non-streaming, ``ttft_ms`` is reported equal to
        ``total_ms`` — TTFT is meaningless without a streaming response.
        """
        try:
            import litellm
        except ImportError as exc:  # pragma: no cover - only triggered if dep is missing
            msg = "litellm is required. Install with: pip install inferencebench-harness"
            raise ClientError(msg) from exc

        messages = build_multimodal_messages(image_path, question)
        kwargs: dict[str, Any] = {
            "model": self._transport.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": False,
            "timeout": self._transport.timeout_s,
            **self._transport.extra_litellm_kwargs,
        }
        if self._transport.api_key:
            kwargs["api_key"] = self._transport.api_key
        if self._transport.base_url:
            kwargs["api_base"] = self._transport.base_url

        t0 = time.perf_counter()
        try:
            response = litellm.completion(**kwargs)
        except Exception as exc:  # rewrap as ClientError per harness contract
            msg = f"multimodal completion failed: {exc}"
            raise ClientError(msg) from exc
        t1 = time.perf_counter()

        try:
            text = response.choices[0].message.content or ""
            finish_reason = response.choices[0].finish_reason or "unknown"
        except (AttributeError, IndexError, TypeError) as exc:
            msg = f"unexpected multimodal response shape: {exc}"
            raise ClientError(msg) from exc

        usage = getattr(response, "usage", None)
        provider_tokens_in = getattr(usage, "prompt_tokens", None) if usage else None
        provider_tokens_out = getattr(usage, "completion_tokens", None) if usage else None
        provider_cost = getattr(response, "_hidden_params", {}).get("response_cost")

        total_ms = (t1 - t0) * 1000.0
        ttft_ms = total_ms  # non-streaming: TTFT == total
        tokens_in = int(provider_tokens_in) if provider_tokens_in is not None else 0
        tokens_out = int(provider_tokens_out) if provider_tokens_out is not None else 0
        if tokens_in == 0 and tokens_out == 0:
            # Conservative word-count fallback when the endpoint omits usage.
            tokens_in = int(len(question.split()) * 1.3)
            tokens_out = int(len(text.split()) * 1.3)
            token_source = "approx-words"
        else:
            token_source = "provider"
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
