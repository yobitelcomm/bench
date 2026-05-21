"""Whisper-compatible HTTP transcription client.

Supports the OpenAI ``/v1/audio/transcriptions`` endpoint shape used by
faster-whisper-server, OpenAI's audio API, and any Whisper-compatible server.
Returns text + per-request timing (TTFT = total for non-streaming responses).

Implementation notes:
    * Stdlib-only (``urllib.request`` + manual ``multipart/form-data`` body
      construction). ``requests`` is intentionally NOT a dependency.
    * Any transport or HTTP-status failure is caught and surfaced as a
      :class:`TranscriptionResult` with ``ok=False`` and a non-empty ``error``
      — the plugin must never crash the run on a single bad request.
    * ``tokens_out`` is approximate (whitespace-split word count). If
      ``tiktoken`` is importable we use the ``cl100k_base`` encoder for a
      tighter count; otherwise we fall back to the word-split estimate.
"""

from __future__ import annotations

import json
import secrets
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Final

_AUDIO_TRANSCRIPTIONS_PATH: Final = "/audio/transcriptions"
_CRLF: Final = b"\r\n"


@dataclass(frozen=True, slots=True)
class TranscriptionResult:
    """Outcome of a single transcription request.

    Shape mirrors the harness :class:`~inferencebench.harness.Sample` numeric
    fields so the plugin can fold these into Sample objects without renaming.
    On failure, ``ok=False`` and ``error`` carries a short diagnostic string;
    every other numeric field is still populated with a sensible default (0.0
    or the measured elapsed time).
    """

    text: str
    total_ms: float
    ttft_ms: float
    tokens_out: int
    cost_usd: float = 0.0
    finish_reason: str = "stop"
    ok: bool = True
    error: str | None = None


def _approx_tokens(text: str) -> int:
    """Approximate token count — tiktoken if available, else whitespace split."""
    try:
        import tiktoken
    except ImportError:
        return len(text.split())
    try:
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        # tiktoken init can fail for many reasons (network, cache, version skew);
        # whitespace tokenization is a safe fallback so we never crash on counts.
        return len(text.split())


def _build_multipart_body(
    *,
    audio_bytes: bytes,
    audio_filename: str,
    model: str,
    language: str | None,
) -> tuple[bytes, str]:
    """Construct a ``multipart/form-data`` body for ``/audio/transcriptions``.

    Returns ``(body_bytes, content_type_header_value)``. The boundary is a
    128-bit random token so it can't collide with any payload contents.
    """
    boundary = "----InferenceBenchBoundary" + secrets.token_hex(16)
    boundary_b = boundary.encode("ascii")
    parts: list[bytes] = []

    def _field(name: str, value: str) -> None:
        parts.append(b"--" + boundary_b + _CRLF)
        parts.append(f'Content-Disposition: form-data; name="{name}"'.encode("ascii") + _CRLF)
        parts.append(_CRLF)
        parts.append(value.encode("utf-8"))
        parts.append(_CRLF)

    # file part — first so faulty servers that stream-parse see it early.
    parts.append(b"--" + boundary_b + _CRLF)
    parts.append(
        (f'Content-Disposition: form-data; name="file"; filename="{audio_filename}"').encode(
            "ascii"
        )
        + _CRLF
    )
    parts.append(b"Content-Type: audio/wav" + _CRLF)
    parts.append(_CRLF)
    parts.append(audio_bytes)
    parts.append(_CRLF)

    _field("model", model)
    if language:
        _field("language", language)

    parts.append(b"--" + boundary_b + b"--" + _CRLF)
    body = b"".join(parts)
    return body, f"multipart/form-data; boundary={boundary}"


def _parse_text(payload: bytes) -> str:
    """Pull ``text`` out of a Whisper-compatible JSON response.

    Both OpenAI and faster-whisper-server return ``{"text": "..."}``. Some
    servers wrap the response in ``{"results": [{"text": "..."}]}`` — we
    accept either. Returns ``""`` if we can't find a text field (the caller
    treats empty text as a soft-fail at the WER layer, not a hard crash).
    """
    try:
        obj = json.loads(payload.decode("utf-8", errors="replace"))
    except json.JSONDecodeError:
        return ""
    if isinstance(obj, dict):
        v = obj.get("text")
        if isinstance(v, str):
            return v
        results = obj.get("results")
        if isinstance(results, list) and results:
            first = results[0]
            if isinstance(first, dict):
                v2 = first.get("text")
                if isinstance(v2, str):
                    return v2
    return ""


def transcribe(
    audio_path: Path,
    *,
    base_url: str,
    model: str,
    api_key: str = "EMPTY",
    language: str | None = None,
    timeout_s: float = 60.0,
) -> TranscriptionResult:
    """POST ``audio_path`` to ``<base_url>/audio/transcriptions`` and parse the reply.

    Args:
        audio_path: Path to the WAV file to upload. Read into memory in full —
            the bundled fixtures are tiny (<= 16 KB) so this is fine; real-world
            users with large clips should stream instead (Phase 2 work).
        base_url: Endpoint base, e.g. ``http://localhost:8000/v1``. The path
            ``/audio/transcriptions`` is appended automatically.
        model: Model identifier sent in the ``model`` form field (e.g.
            ``whisper-1``, ``Systran/faster-whisper-large-v3``).
        api_key: Bearer token. Defaults to ``"EMPTY"`` (the convention for
            local self-hosted servers that don't authenticate).
        language: Optional BCP-47 / ISO-639 language hint passed verbatim to
            the server. Omitted from the multipart body when ``None``.
        timeout_s: Per-request timeout in seconds.

    Returns:
        :class:`TranscriptionResult`. On any failure (read error, HTTP error,
        bad JSON, timeout, connection reset) returns ``ok=False`` with a
        non-empty ``error`` string. ``total_ms`` and ``ttft_ms`` are always
        the measured wall-clock for the attempt, even on failure.
    """
    start = time.perf_counter()
    try:
        audio_bytes = audio_path.read_bytes()
    except OSError as exc:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return TranscriptionResult(
            text="",
            total_ms=elapsed_ms,
            ttft_ms=elapsed_ms,
            tokens_out=0,
            ok=False,
            error=f"audio read failed: {exc}",
        )

    body, content_type = _build_multipart_body(
        audio_bytes=audio_bytes,
        audio_filename=audio_path.name,
        model=model,
        language=language,
    )

    url = base_url.rstrip("/") + _AUDIO_TRANSCRIPTIONS_PATH
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", content_type)
    req.add_header("Content-Length", str(len(body)))
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")

    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        text = _parse_text(raw)
        return TranscriptionResult(
            text=text,
            total_ms=elapsed_ms,
            ttft_ms=elapsed_ms,
            tokens_out=_approx_tokens(text),
        )
    except urllib.error.HTTPError as exc:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return TranscriptionResult(
            text="",
            total_ms=elapsed_ms,
            ttft_ms=elapsed_ms,
            tokens_out=0,
            ok=False,
            error=f"HTTP {exc.code}: {exc.reason}",
        )
    except urllib.error.URLError as exc:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return TranscriptionResult(
            text="",
            total_ms=elapsed_ms,
            ttft_ms=elapsed_ms,
            tokens_out=0,
            ok=False,
            error=f"connection failed: {exc.reason}",
        )
    except (TimeoutError, OSError) as exc:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return TranscriptionResult(
            text="",
            total_ms=elapsed_ms,
            ttft_ms=elapsed_ms,
            tokens_out=0,
            ok=False,
            error=f"transport error: {exc}",
        )
