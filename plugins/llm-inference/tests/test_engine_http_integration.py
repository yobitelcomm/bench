"""End-to-end engine adapter tests against a stdlib HTTP stub server.

The unit tests in ``test_<engine>_engine.py`` exercise each adapter's logic
by monkeypatching :func:`urllib.request.urlopen`. That misses dispatch-level
bugs — URL composition, header handling, real socket I/O — because the
patched function is exactly the thing under test.

This module spins up a ``ThreadingHTTPServer`` on an ephemeral port per
test, points the adapter at it, and verifies the full request/response
round-trip:

- ``probe()`` returns the expected version string when the stub responds 200.
- The stub's request log shows the adapter hit the expected probe path.
- ``probe()`` raises :class:`EngineUnavailableError` on a 500 response.
- ``probe()`` raises :class:`EngineUnavailableError` on connection refused.
- ``build_client()`` produces a :class:`ModelClient` with the right model
  id + base_url.

Stub server constraints (deliberately minimal):

- Pure stdlib (``http.server.ThreadingHTTPServer`` + ``BaseHTTPRequestHandler``)
  — no ``responses`` / ``httpx-mock`` deps.
- Thread-safe — each test owns its own port + thread + log.
- Cleanup via fixture finalizer (``server.shutdown()``).
"""

from __future__ import annotations

import json
import socket
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, ClassVar

import pytest

from inferencebench_llm import EngineKind, RunContext
from inferencebench_llm.engines import (
    Engine,
    EngineUnavailableError,
    LlamaCppEngine,
    MLXEngine,
    SGLangEngine,
    TRTLLMEngine,
    VLLMEngine,
)


# --------------------------------------------------------------------------- #
# Stub HTTP server                                                            #
# --------------------------------------------------------------------------- #
class HTTPStubServer:
    """Per-test ephemeral HTTP stub.

    Routes are a dict keyed on ``(method, path)`` with a tuple value
    ``(status, headers, body)``. Any request whose path doesn't match a
    configured route returns 404. Every request (whether matched or not)
    is appended to ``requests`` as ``(method, path, body_bytes)`` for
    later inspection.
    """

    def __init__(self) -> None:
        # ``threading.Lock`` is enough — request volume in tests is low and
        # contention is bounded by the single ServeThread we spawn.
        self._lock = threading.Lock()
        self.routes: dict[tuple[str, str], tuple[int, dict[str, str], bytes]] = {}
        self.requests: list[tuple[str, str, bytes]] = []
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self.port: int = 0

    def set_route(
        self,
        method: str,
        path: str,
        *,
        status: int = 200,
        headers: dict[str, str] | None = None,
        body: bytes | str | dict[str, Any] | None = None,
    ) -> None:
        """Register a canned response. ``body`` accepts bytes / str / dict (JSON)."""
        if isinstance(body, dict):
            payload = json.dumps(body).encode("utf-8")
            hdrs = dict(headers or {})
            hdrs.setdefault("Content-Type", "application/json")
        elif isinstance(body, str):
            payload = body.encode("utf-8")
            hdrs = dict(headers or {})
        elif body is None:
            payload = b""
            hdrs = dict(headers or {})
        else:
            payload = body
            hdrs = dict(headers or {})
        with self._lock:
            self.routes[(method.upper(), path)] = (status, hdrs, payload)

    def start(self) -> None:
        """Bind to an ephemeral port and serve in a background thread."""
        handler = _make_handler(self)
        # Port 0 → OS picks an unused port; we read it back after binding.
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.port = self._server.server_address[1]
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name=f"HTTPStubServer-{self.port}",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Shut the server down and join the background thread."""
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    # Internal — invoked from the handler.
    def _record_request(self, method: str, path: str, body: bytes) -> None:
        with self._lock:
            self.requests.append((method, path, body))

    def _lookup_route(self, method: str, path: str) -> tuple[int, dict[str, str], bytes] | None:
        with self._lock:
            return self.routes.get((method.upper(), path))


def _make_handler(stub: HTTPStubServer) -> type[BaseHTTPRequestHandler]:
    """Build a per-stub ``BaseHTTPRequestHandler`` subclass closed over ``stub``."""

    class _Handler(BaseHTTPRequestHandler):
        # Suppress per-request stderr logging — the test runner is noisy enough.
        stub_ref: ClassVar[HTTPStubServer] = stub

        def log_message(self, format: str, *args: Any) -> None:
            return

        def _read_body(self) -> bytes:
            length = int(self.headers.get("Content-Length") or "0")
            return self.rfile.read(length) if length > 0 else b""

        def _serve(self, method: str) -> None:
            body = self._read_body()
            self.stub_ref._record_request(method, self.path, body)
            route = self.stub_ref._lookup_route(method, self.path)
            if route is None:
                # send_response_only + manual Date/Content-Length so the
                # stdlib's auto-injected ``Server`` header doesn't clobber
                # routes that need a specific ``Server`` value (TRT-LLM).
                self.send_response_only(404, "Not Found")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            status, headers, payload = route
            self.send_response_only(status)
            # Caller-supplied headers come first so a custom ``Server`` value
            # wins over the stdlib's auto-injected one. (We bypass
            # send_response entirely so no Server/Date header is auto-added.)
            for k, v in headers.items():
                self.send_header(k, v)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            if payload:
                self.wfile.write(payload)

        def do_GET(self) -> None:
            self._serve("GET")

        def do_POST(self) -> None:
            self._serve("POST")

    return _Handler


def _find_free_port() -> int:
    """Return a currently-free TCP port for the connection-refused tests."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #
@pytest.fixture
def stub_server() -> Iterator[HTTPStubServer]:
    """Per-test stub server. Finalizer guarantees shutdown + thread join."""
    stub = HTTPStubServer()
    stub.start()
    try:
        yield stub
    finally:
        stub.stop()


# --------------------------------------------------------------------------- #
# Per-engine canned responses                                                 #
# --------------------------------------------------------------------------- #
# Each entry: (engine_cls, expected_probe_path, expected_version, routes).
# ``routes`` is a list of ``(method, path, status, headers, body)`` tuples to
# preconfigure on the stub before the call.
def _vllm_routes() -> list[tuple[str, str, int, dict[str, str], dict[str, Any]]]:
    """vLLM hits ``/v1/models`` and falls back to ``/version`` for the version."""
    return [
        (
            "GET",
            "/v1/models",
            200,
            {},
            {"object": "list", "data": [{"id": "meta-llama/Llama-3.1-8B-Instruct"}]},
        ),
        ("GET", "/version", 200, {}, {"version": "0.7.2"}),
    ]


def _sglang_routes() -> list[tuple[str, str, int, dict[str, str], dict[str, Any]]]:
    """SGLang exposes ``/get_server_info`` with a top-level ``version``."""
    return [
        (
            "GET",
            "/get_server_info",
            200,
            {},
            {"version": "0.4.5", "server_args": {"model_path": "Qwen/Qwen2.5-7B-Instruct"}},
        ),
    ]


def _llamacpp_routes() -> list[tuple[str, str, int, dict[str, str], dict[str, Any]]]:
    """llama.cpp surfaces ``/props`` with a ``system_info`` build-context string."""
    return [
        (
            "GET",
            "/props",
            200,
            {},
            {
                "system_info": (
                    "llama.cpp AVX2 1 | AVX_VNNI 0 | AVX512 0 | AVX512_VBMI 0 | BUILD_TIME 2026-04"
                ),
                "model_path": "/models/foo.gguf",
                "n_ctx": 4096,
            },
        ),
    ]


def _trtllm_routes() -> list[tuple[str, str, int, dict[str, str], dict[str, Any]]]:
    """TRT-LLM surfaces ``/health/load`` and carries the version in ``Server:``."""
    return [
        (
            "GET",
            "/health/load",
            200,
            {"Server": "trtllm-0.13.0"},
            {"queue_length": 0, "kv_cache_used_ratio": 0.0},
        ),
    ]


def _mlx_routes() -> list[tuple[str, str, int, dict[str, str], dict[str, Any]]]:
    """mlx_lm.server only exposes ``/v1/models``; probe returns ``"unknown"``."""
    return [
        (
            "GET",
            "/v1/models",
            200,
            {},
            {"object": "list", "data": [{"id": "mlx-community/qwen2.5-7b-instruct"}]},
        ),
    ]


# (label, engine_cls, engine_kind, expected_probe_path, expected_version, routes)
ENGINE_CASES: list[
    tuple[
        str,
        type[Engine],
        EngineKind,
        str,
        str,
        list[tuple[str, str, int, dict[str, str], dict[str, Any]]],
    ]
] = [
    ("vllm", VLLMEngine, EngineKind.VLLM, "/v1/models", "0.7.2", _vllm_routes()),
    (
        "sglang",
        SGLangEngine,
        EngineKind.SGLANG,
        "/get_server_info",
        "0.4.5",
        _sglang_routes(),
    ),
    (
        "llamacpp",
        LlamaCppEngine,
        EngineKind.LLAMACPP,
        "/props",
        "llama.cpp",
        _llamacpp_routes(),
    ),
    (
        "trtllm",
        TRTLLMEngine,
        EngineKind.TRTLLM,
        "/health/load",
        "0.13.0",
        _trtllm_routes(),
    ),
    ("mlx", MLXEngine, EngineKind.MLX, "/v1/models", "unknown", _mlx_routes()),
]


def _configure_stub(
    stub: HTTPStubServer,
    routes: list[tuple[str, str, int, dict[str, str], dict[str, Any]]],
) -> None:
    for method, path, status, headers, body in routes:
        stub.set_route(method, path, status=status, headers=headers, body=body)


# --------------------------------------------------------------------------- #
# Round-trip success                                                          #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("label", "engine_cls", "engine_kind", "probe_path", "expected_version", "routes"),
    ENGINE_CASES,
    ids=[c[0] for c in ENGINE_CASES],
)
def test_engine_probe_round_trip(
    stub_server: HTTPStubServer,
    label: str,
    engine_cls: type[Engine],
    engine_kind: EngineKind,
    probe_path: str,
    expected_version: str,
    routes: list[tuple[str, str, int, dict[str, str], dict[str, Any]]],
) -> None:
    """Each engine returns the right version + hits the documented probe path."""
    _configure_stub(stub_server, routes)
    ctx = RunContext(
        model_id="meta-llama/Llama-3.1-8B-Instruct",
        engine_kind=engine_kind,
        base_url=f"{stub_server.base_url}/v1",
        output_dir=Path("/tmp/bench"),
    )
    engine = engine_cls()
    version = engine.probe(ctx)
    # llama.cpp returns the first 60 chars of system_info — we just assert
    # the canonical prefix is there; the rest is build-stamp.
    if label == "llamacpp":
        assert version.startswith("llama.cpp")
    else:
        assert version == expected_version

    paths_hit = [p for _m, p, _b in stub_server.requests]
    assert probe_path in paths_hit, (
        f"{label}: expected probe path {probe_path!r} not in {paths_hit!r}"
    )


@pytest.mark.parametrize(
    ("label", "engine_cls", "engine_kind", "probe_path", "expected_version", "routes"),
    ENGINE_CASES,
    ids=[c[0] for c in ENGINE_CASES],
)
def test_engine_build_client_after_probe(
    stub_server: HTTPStubServer,
    label: str,
    engine_cls: type[Engine],
    engine_kind: EngineKind,
    probe_path: str,
    expected_version: str,
    routes: list[tuple[str, str, int, dict[str, str], dict[str, Any]]],
) -> None:
    """build_client() returns a ModelClient with the right model + base_url."""
    _configure_stub(stub_server, routes)
    ctx = RunContext(
        model_id="meta-llama/Llama-3.1-8B-Instruct",
        engine_kind=engine_kind,
        base_url=f"{stub_server.base_url}/v1",
        output_dir=Path("/tmp/bench"),
    )
    engine = engine_cls()
    engine.probe(ctx)  # smoke-check the round-trip before client construction.
    client = engine.build_client(ctx)
    # All five adapters emit LiteLLM-routed ``openai/<model>``.
    assert client.model == "openai/meta-llama/Llama-3.1-8B-Instruct"
    assert client.base_url == f"{stub_server.base_url}/v1"
    # Unused argument silencer for parametrize tuple signature.
    _ = probe_path
    _ = expected_version
    _ = label


# --------------------------------------------------------------------------- #
# Error paths                                                                 #
# --------------------------------------------------------------------------- #
def _stub_returns_500(stub: HTTPStubServer, paths: list[str]) -> None:
    """Wire every probe path the engines might hit to a 500."""
    for path in paths:
        stub.set_route("GET", path, status=500, body=b"server error")


@pytest.mark.parametrize(
    ("label", "engine_cls", "engine_kind"),
    [(c[0], c[1], c[2]) for c in ENGINE_CASES],
    ids=[c[0] for c in ENGINE_CASES],
)
def test_engine_probe_raises_on_500(
    stub_server: HTTPStubServer,
    label: str,
    engine_cls: type[Engine],
    engine_kind: EngineKind,
) -> None:
    """A 500 response on every documented probe path → ``EngineUnavailableError``."""
    # Wire all of the candidate probe paths (primary + fallbacks) to 500 so
    # each engine has nowhere to fall back to.
    _stub_returns_500(
        stub_server,
        ["/v1/models", "/get_server_info", "/props", "/health/load", "/version"],
    )
    ctx = RunContext(
        model_id="meta-llama/Llama-3.1-8B-Instruct",
        engine_kind=engine_kind,
        base_url=f"{stub_server.base_url}/v1",
        output_dir=Path("/tmp/bench"),
    )
    engine = engine_cls()
    with pytest.raises(EngineUnavailableError):
        engine.probe(ctx)
    _ = label


@pytest.mark.parametrize(
    ("label", "engine_cls", "engine_kind"),
    [(c[0], c[1], c[2]) for c in ENGINE_CASES],
    ids=[c[0] for c in ENGINE_CASES],
)
def test_engine_probe_raises_on_connection_refused(
    label: str,
    engine_cls: type[Engine],
    engine_kind: EngineKind,
) -> None:
    """Pointing at a closed port → ``EngineUnavailableError``."""
    port = _find_free_port()
    ctx = RunContext(
        model_id="meta-llama/Llama-3.1-8B-Instruct",
        engine_kind=engine_kind,
        base_url=f"http://127.0.0.1:{port}/v1",
        output_dir=Path("/tmp/bench"),
    )
    engine = engine_cls()
    with pytest.raises(EngineUnavailableError):
        engine.probe(ctx)
    _ = label
