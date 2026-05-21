"""``bench server`` — minimal stdlib HTTP server for envelope ingestion.

A central host runs ``bench server --port 8080 --store ./envelopes/`` and
distributed runners (cron jobs, CI fleets, contributors) POST signed envelopes
to it. The server validates each envelope's schema + signature, drops it on
disk under ``<store>/<content_hash[:12]>.json``, and exposes a read endpoint.

Phase 1 ships the minimal version with stdlib only — no Flask / FastAPI /
Starlette. The point is dependency-free deployability for early adopters; once
the workflow firms up we can graduate to a real framework.
"""

from __future__ import annotations

import json
import sys
import threading
from collections.abc import Callable
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import metadata as _metadata
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console

from inferencebench.envelope import Envelope, verify_envelope

console = Console()
err_console = Console(stderr=True)


def _bench_version() -> str:
    """Return the installed ``inferencebench`` distribution version."""
    try:
        return _metadata.version("inferencebench")
    except _metadata.PackageNotFoundError:
        return "0.0.0"


# --------------------------------------------------------------------------- #
# Command                                                                     #
# --------------------------------------------------------------------------- #
def server(
    port: Annotated[
        int,
        typer.Option("--port", help="TCP port to bind. Default 8080."),
    ] = 8080,
    host: Annotated[
        str,
        typer.Option("--host", help="Host/interface to bind. Default 127.0.0.1."),
    ] = "127.0.0.1",
    store: Annotated[
        Path,
        typer.Option(
            "--store",
            help="Directory to write accepted envelopes into. Created if missing.",
        ),
    ] = Path("./envelopes/"),
    dev_public_key: Annotated[
        Path | None,
        typer.Option(
            "--dev-public-key",
            help=(
                "Path to ed25519 public key. Required to accept dev-key envelopes; "
                "without it, dev-key POSTs are rejected with 401."
            ),
        ),
    ] = None,
) -> None:
    """Run the envelope ingestion HTTP server. Blocking; Ctrl-C to stop."""
    store.mkdir(parents=True, exist_ok=True)

    httpd = make_server(
        host=host,
        port=port,
        store=store,
        dev_public_key=dev_public_key,
    )

    address = httpd.server_address
    actual_host = address[0] if isinstance(address[0], str) else address[0].decode()
    actual_port = address[1]
    console.print(
        f"[bold green]bench server[/bold green] listening on http://{actual_host}:{actual_port}"
    )
    console.print(f"  store:           {store.resolve()}")
    pub_label = (
        str(dev_public_key.resolve())
        if dev_public_key
        else "(none — dev-key envelopes will be rejected)"
    )
    console.print(f"  dev_public_key:  {pub_label}")
    console.print("  endpoints:")
    console.print("    GET  /health")
    console.print("    GET  /envelopes")
    console.print("    GET  /envelopes/<content_hash>")
    console.print("    POST /envelopes")
    console.print("Press Ctrl-C to stop.")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        console.print("\n[yellow]Shutting down…[/yellow]")
    finally:
        httpd.shutdown()
        httpd.server_close()


# --------------------------------------------------------------------------- #
# Server factory                                                              #
# --------------------------------------------------------------------------- #
class _EnvelopeServer(ThreadingHTTPServer):
    """Threading HTTP server carrying ingestion config."""

    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        handler_class: type[BaseHTTPRequestHandler],
        *,
        store: Path,
        dev_public_key: Path | None,
    ) -> None:
        super().__init__(server_address, handler_class)
        self.store: Path = store
        self.dev_public_key: Path | None = dev_public_key
        self._lock = threading.Lock()

    def write_envelope(self, envelope: Envelope, raw: bytes) -> Path:
        """Persist an accepted envelope under ``<store>/<hash[:12]>.json``."""
        target = self.store / f"{envelope.content_hash()[:12]}.json"
        with self._lock:
            target.write_bytes(raw)
        return target


def make_server(
    *,
    host: str,
    port: int,
    store: Path,
    dev_public_key: Path | None,
) -> _EnvelopeServer:
    """Construct an ``_EnvelopeServer`` ready to ``serve_forever()``.

    The store directory is created if missing so callers (including tests)
    do not have to mkdir it themselves. Exposed as a helper so tests can spin
    the server up on an ephemeral port in a worker thread.
    """
    store.mkdir(parents=True, exist_ok=True)
    return _EnvelopeServer(
        (host, port),
        _Handler,
        store=store,
        dev_public_key=dev_public_key,
    )


# --------------------------------------------------------------------------- #
# Request handler                                                             #
# --------------------------------------------------------------------------- #
class _Handler(BaseHTTPRequestHandler):
    """stdlib-only HTTP handler for the envelope ingestion API."""

    server: _EnvelopeServer  # narrowed type for mypy

    # ----- response helpers -------------------------------------------------
    def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, sort_keys=True).encode("utf-8")
        self.send_response(int(status))
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_raw_json(self, status: HTTPStatus, body: bytes) -> None:
        self.send_response(int(status))
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # ----- structured logging to stderr ------------------------------------
    def log_message(self, format: str, *args: object) -> None:  # noqa: ARG002
        """Log a one-line request summary to stderr.

        Overrides BaseHTTPRequestHandler.log_message; the ``format`` parameter
        is part of the upstream contract but we render our own line.
        """
        status = args[1] if len(args) > 1 else "?"
        sys.stderr.write(
            f"[bench server] {self.address_string()} - {self.command} {self.path} -> {status}\n"
        )
        sys.stderr.flush()

    # ----- routing ----------------------------------------------------------
    def do_GET(self) -> None:
        """Route GET requests to the matching handler."""
        route_get: dict[str, Callable[[], None]] = {
            "/health": self._handle_health,
            "/envelopes": self._handle_list_envelopes,
        }
        path = self.path.split("?", 1)[0]
        if path in route_get:
            route_get[path]()
            return
        if path.startswith("/envelopes/"):
            self._handle_get_envelope(path[len("/envelopes/") :])
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def do_POST(self) -> None:
        """Route POST requests to the matching handler."""
        path = self.path.split("?", 1)[0]
        if path == "/envelopes":
            self._handle_post_envelope()
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    # ----- handlers ---------------------------------------------------------
    def _handle_health(self) -> None:
        store_count = sum(1 for _ in self.server.store.glob("*.json"))
        self._send_json(
            HTTPStatus.OK,
            {
                "status": "ok",
                "version": _bench_version(),
                "store_count": store_count,
            },
        )

    def _handle_list_envelopes(self) -> None:
        entries: list[dict[str, str]] = []
        for path in sorted(self.server.store.glob("*.json")):
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                envelope = Envelope.model_validate(raw)
            except (OSError, ValueError):
                # Garbled or non-envelope file in the store — skip silently so
                # one bad file does not poison the whole listing.
                continue
            entries.append(
                {
                    "content_hash": envelope.content_hash(),
                    "suite_id": envelope.suite_id,
                    "model_id": envelope.model.id,
                }
            )
        self._send_json(HTTPStatus.OK, {"entries": entries})

    def _handle_get_envelope(self, hash_fragment: str) -> None:
        if not hash_fragment or "/" in hash_fragment or ".." in hash_fragment:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "invalid hash"})
            return
        candidate = self.server.store / f"{hash_fragment[:12]}.json"
        if not candidate.exists():
            self._send_json(
                HTTPStatus.NOT_FOUND,
                {"error": f"envelope {hash_fragment[:12]} not found"},
            )
            return
        try:
            body = candidate.read_bytes()
        except OSError as exc:
            self._send_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"error": f"failed to read envelope: {exc}"},
            )
            return
        self._send_raw_json(HTTPStatus.OK, body)

    def _handle_post_envelope(self) -> None:
        length_header = self.headers.get("Content-Length")
        try:
            length = int(length_header or "0")
        except ValueError:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "invalid Content-Length"})
            return
        if length <= 0:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "empty body"})
            return

        raw_body = self.rfile.read(length)

        try:
            parsed = json.loads(raw_body.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": f"invalid JSON: {exc}"})
            return

        try:
            envelope = Envelope.model_validate(parsed)
        except ValueError as exc:
            # pydantic.ValidationError is a subclass of ValueError — catching
            # ValueError covers both pydantic failures and any envelope-level
            # cross-field validators that raise plain ValueError.
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {"error": f"envelope schema validation failed: {exc}"},
            )
            return

        if envelope.signature is None:
            self._send_json(
                HTTPStatus.UNAUTHORIZED,
                {"error": "envelope has no signature"},
            )
            return

        if envelope.signature.method == "dev-key" and self.server.dev_public_key is None:
            self._send_json(
                HTTPStatus.UNAUTHORIZED,
                {
                    "error": (
                        "server has no --dev-public-key configured; dev-key envelopes are rejected"
                    )
                },
            )
            return

        try:
            result = verify_envelope(envelope, dev_public_key_path=self.server.dev_public_key)
        except (ValueError, OSError) as exc:
            self._send_json(
                HTTPStatus.UNAUTHORIZED,
                {"error": f"signature verification raised: {exc}"},
            )
            return

        if not result.ok:
            self._send_json(
                HTTPStatus.UNAUTHORIZED,
                {"error": f"signature verification failed: {result.reason}"},
            )
            return

        stored_path = self.server.write_envelope(envelope, raw_body)
        self._send_json(
            HTTPStatus.CREATED,
            {
                "content_hash": envelope.content_hash(),
                "stored_at": str(stored_path),
            },
        )
