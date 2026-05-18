"""Tests for the Whisper-compatible HTTP transcription client.

A stub server (stdlib :class:`http.server.ThreadingHTTPServer`) stands in for
faster-whisper-server / OpenAI's audio API. The client must:
    * POST to ``/audio/transcriptions`` and parse a ``{"text": "..."}`` reply.
    * Surface 5xx HTTP failures as ``ok=False`` results (no crash).
    * Surface transport failures (connection reset, timeout) similarly.
    * Send a real multipart body with ``file``, ``model``, and (optionally)
      ``language`` parts.

No real Whisper server is contacted.
"""

from __future__ import annotations

import socket
import threading
import time
import wave
from collections.abc import Iterator
from contextlib import closing, contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from inferencebench_voice.audio_client import transcribe


# --------------------------------------------------------------------------- #
# Stub server plumbing                                                        #
# --------------------------------------------------------------------------- #
def _free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class _StubState:
    """Holds the per-test stub configuration and captured request data."""

    def __init__(self) -> None:
        self.mode: str = "ok"  # "ok" | "500" | "close_mid"
        self.response_text: str = "hello world"
        self.last_path: str | None = None
        self.last_method: str | None = None
        self.last_content_type: str | None = None
        self.last_body: bytes = b""


def _make_handler(state: _StubState) -> type[BaseHTTPRequestHandler]:
    class _Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            # Silence test-server stderr noise.
            return

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length) if length else b""
            state.last_path = self.path
            state.last_method = "POST"
            state.last_content_type = self.headers.get("Content-Type")
            state.last_body = body

            if state.mode == "500":
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"error":"boom"}')
                return
            if state.mode == "close_mid":
                # Slam the connection without sending response headers.
                try:
                    self.connection.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                self.connection.close()
                return

            payload = ('{"text":"' + state.response_text + '"}').encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    return _Handler


@contextmanager
def _stub_server(state: _StubState) -> Iterator[str]:
    port = _free_port()
    handler_cls = _make_handler(state)
    httpd = ThreadingHTTPServer(("127.0.0.1", port), handler_cls)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}/v1"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=2.0)


@pytest.fixture
def wav_path(tmp_path: Path) -> Path:
    """Tiny 16 kHz mono PCM WAV: 0.05 s of silence, ~1.6 KB."""
    path = tmp_path / "tiny.wav"
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16_000)
        wf.writeframes(b"\x00\x00" * 800)
    return path


# --------------------------------------------------------------------------- #
# Happy path                                                                  #
# --------------------------------------------------------------------------- #
def test_transcribe_parses_text_field(wav_path: Path) -> None:
    state = _StubState()
    state.response_text = "hello world"
    with _stub_server(state) as base_url:
        result = transcribe(
            wav_path,
            base_url=base_url,
            model="whisper-1",
            timeout_s=5.0,
        )
    assert result.ok is True
    assert result.error is None
    assert result.text == "hello world"
    assert result.tokens_out >= 2
    assert result.total_ms > 0
    assert result.ttft_ms == result.total_ms
    assert result.finish_reason == "stop"


def test_transcribe_posts_to_audio_transcriptions(wav_path: Path) -> None:
    state = _StubState()
    with _stub_server(state) as base_url:
        transcribe(wav_path, base_url=base_url, model="whisper-1", timeout_s=5.0)
    assert state.last_path == "/v1/audio/transcriptions"
    assert state.last_method == "POST"
    assert state.last_content_type is not None
    assert state.last_content_type.startswith("multipart/form-data; boundary=")


def test_multipart_body_carries_file_and_model_parts(wav_path: Path) -> None:
    state = _StubState()
    with _stub_server(state) as base_url:
        transcribe(
            wav_path,
            base_url=base_url,
            model="whisper-large-v3",
            timeout_s=5.0,
        )
    body = state.last_body
    assert b'name="file"' in body
    assert wav_path.name.encode() in body
    assert b'name="model"' in body
    assert b"whisper-large-v3" in body
    # No language form field when the caller doesn't pass one.
    assert b'name="language"' not in body


def test_multipart_body_includes_language_when_supplied(wav_path: Path) -> None:
    state = _StubState()
    with _stub_server(state) as base_url:
        transcribe(
            wav_path,
            base_url=base_url,
            model="whisper-1",
            language="en",
            timeout_s=5.0,
        )
    body = state.last_body
    assert b'name="language"' in body
    # The value sits on its own line after a CRLF blank.
    assert b"\r\nen\r\n" in body


# --------------------------------------------------------------------------- #
# Failure modes                                                               #
# --------------------------------------------------------------------------- #
def test_transcribe_returns_ok_false_on_http_500(wav_path: Path) -> None:
    state = _StubState()
    state.mode = "500"
    with _stub_server(state) as base_url:
        result = transcribe(
            wav_path,
            base_url=base_url,
            model="whisper-1",
            timeout_s=5.0,
        )
    assert result.ok is False
    assert result.error is not None
    assert "500" in result.error
    assert result.text == ""
    assert result.tokens_out == 0


def test_transcribe_returns_ok_false_on_mid_request_disconnect(
    wav_path: Path,
) -> None:
    state = _StubState()
    state.mode = "close_mid"
    with _stub_server(state) as base_url:
        result = transcribe(
            wav_path,
            base_url=base_url,
            model="whisper-1",
            timeout_s=2.0,
        )
    assert result.ok is False
    assert result.error is not None
    assert result.error != ""
    assert result.text == ""


def test_transcribe_returns_ok_false_when_server_unreachable(
    wav_path: Path,
) -> None:
    # Pick a free port but DON'T start a server on it — connection should refuse.
    port = _free_port()
    start = time.perf_counter()
    result = transcribe(
        wav_path,
        base_url=f"http://127.0.0.1:{port}/v1",
        model="whisper-1",
        timeout_s=2.0,
    )
    elapsed = time.perf_counter() - start
    assert result.ok is False
    assert result.error is not None
    assert result.error != ""
    # Connection refusal returns fast; the test should not have waited the full timeout.
    assert elapsed < 5.0


def test_transcribe_handles_missing_audio_file(tmp_path: Path) -> None:
    missing = tmp_path / "nope.wav"
    state = _StubState()
    with _stub_server(state) as base_url:
        result = transcribe(
            missing,
            base_url=base_url,
            model="whisper-1",
            timeout_s=2.0,
        )
    assert result.ok is False
    assert result.error is not None
    assert "audio read failed" in result.error
