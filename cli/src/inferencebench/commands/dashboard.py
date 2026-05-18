"""``bench dashboard`` — local HTTP server serving a live leaderboard.

Today ``bench leaderboard --build`` produces a static site that has to be
re-rendered after every new envelope. ``bench dashboard`` serves the same data
over HTTP from an envelope directory, automatically rescanning + re-rendering
on each request whenever the cached render exceeds ``--rebuild-interval-s``.

Ideal during active development sessions: drop another envelope into the
watched directory, refresh the page, see the new row. Stdlib only — same
``http.server.ThreadingHTTPServer`` pattern as ``bench server``.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import threading
import time
from datetime import UTC, datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console

console = Console()
err_console = Console(stderr=True)

_CONTENT_TYPES: dict[str, str] = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
}


# --------------------------------------------------------------------------- #
# Command                                                                     #
# --------------------------------------------------------------------------- #
def dashboard(
    envelopes_dir: Annotated[
        Path,
        typer.Option(
            "--envelopes",
            help="Directory of signed envelope JSON files to serve.",
        ),
    ],
    port: Annotated[
        int,
        typer.Option("--port", help="TCP port to bind. Default 8090."),
    ] = 8090,
    host: Annotated[
        str,
        typer.Option("--host", help="Host/interface to bind. Default 127.0.0.1."),
    ] = "127.0.0.1",
    rebuild_interval_s: Annotated[
        float,
        typer.Option(
            "--rebuild-interval-s",
            help="Max age (seconds) of the cached render before a fresh scan.",
        ),
    ] = 2.0,
) -> None:
    """Serve the leaderboard over HTTP with live rescanning of envelopes."""
    if not envelopes_dir.exists() or not envelopes_dir.is_dir():
        err_console.print(
            f"[red]Envelopes directory not found:[/red] {envelopes_dir}"
        )
        raise typer.Exit(code=2)

    try:
        from inferencebench_leaderboard import render_site  # noqa: F401
    except ImportError as exc:
        err_console.print(
            "[red]inferencebench-leaderboard is not installed.[/red] "
            "Install it: [bold]pip install inferencebench-leaderboard[/bold]"
        )
        raise typer.Exit(code=2) from exc

    httpd = make_dashboard_server(
        host=host,
        port=port,
        envelopes_dir=envelopes_dir,
        rebuild_interval_s=rebuild_interval_s,
    )
    address = httpd.server_address
    actual_host = address[0] if isinstance(address[0], str) else address[0].decode()
    actual_port = address[1]

    console.print(
        f"[bold green]bench dashboard[/bold green] listening on "
        f"http://{actual_host}:{actual_port}"
    )
    console.print(f"  envelopes:          {envelopes_dir.resolve()}")
    console.print(f"  rebuild_interval_s: {rebuild_interval_s}")
    console.print("  endpoints:")
    console.print("    GET  /")
    console.print("    GET  /<suite>/")
    console.print("    GET  /data/leaderboard.json")
    console.print("    GET  /__health__")
    console.print("    GET  /__rescan__")
    console.print("Press Ctrl-C to stop.")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        console.print("\n[yellow]Shutting down…[/yellow]")
    finally:
        httpd.shutdown()
        httpd.server_close()
        httpd.cleanup()


# --------------------------------------------------------------------------- #
# Server                                                                      #
# --------------------------------------------------------------------------- #
class _DashboardServer(ThreadingHTTPServer):
    """Threading HTTP server backing ``bench dashboard``.

    Owns the ephemeral cache directory, the cached render metadata, and the
    rebuild lock. The handler reads from ``cache_dir`` and may trigger a
    refresh via :meth:`maybe_rebuild` / :meth:`force_rebuild`.
    """

    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        handler_class: type[BaseHTTPRequestHandler],
        *,
        envelopes_dir: Path,
        rebuild_interval_s: float,
    ) -> None:
        super().__init__(server_address, handler_class)
        self.envelopes_dir: Path = envelopes_dir
        self.rebuild_interval_s: float = rebuild_interval_s
        self._cache_root = Path(
            tempfile.mkdtemp(prefix="bench-dashboard-")
        )
        self.cache_dir: Path = self._cache_root / "site"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._last_render_monotonic: float | None = None
        self.last_render_iso: str = ""
        self.envelopes_count: int = 0

    # ----- rendering ------------------------------------------------------- #
    def _render_now_locked(self) -> None:
        """Re-render into ``cache_dir``. Caller must hold ``self._lock``."""
        from inferencebench_leaderboard import render_site

        result = render_site(
            self.envelopes_dir, self.cache_dir, base_url="/"
        )
        self._last_render_monotonic = time.monotonic()
        self.last_render_iso = datetime.now(UTC).isoformat()
        self.envelopes_count = result.envelopes_loaded

    def maybe_rebuild(self) -> None:
        """Rebuild if the cache is older than ``rebuild_interval_s``."""
        with self._lock:
            now = time.monotonic()
            age = (
                float("inf")
                if self._last_render_monotonic is None
                else now - self._last_render_monotonic
            )
            if age >= self.rebuild_interval_s:
                self._render_now_locked()

    def force_rebuild(self) -> None:
        """Re-render unconditionally."""
        with self._lock:
            self._render_now_locked()

    # ----- teardown -------------------------------------------------------- #
    def cleanup(self) -> None:
        """Remove the ephemeral cache directory."""
        shutil.rmtree(self._cache_root, ignore_errors=True)


def make_dashboard_server(
    *,
    host: str,
    port: int,
    envelopes_dir: Path,
    rebuild_interval_s: float,
) -> _DashboardServer:
    """Construct a ``_DashboardServer`` ready to ``serve_forever()``.

    The initial render is performed eagerly so the first request can be served
    from cache without surprise latency. Exposed for tests so they can spin
    the dashboard up on an ephemeral port in a worker thread.
    """
    server = _DashboardServer(
        (host, port),
        _Handler,
        envelopes_dir=envelopes_dir,
        rebuild_interval_s=rebuild_interval_s,
    )
    server.force_rebuild()
    return server


# --------------------------------------------------------------------------- #
# Request handler                                                             #
# --------------------------------------------------------------------------- #
class _Handler(BaseHTTPRequestHandler):
    """stdlib-only HTTP handler serving the cached leaderboard site."""

    server: _DashboardServer  # narrowed type for mypy

    # ----- response helpers ------------------------------------------------ #
    def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, sort_keys=True).encode("utf-8")
        self.send_response(int(status))
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, status: HTTPStatus, path: Path) -> None:
        try:
            body = path.read_bytes()
        except OSError as exc:
            self._send_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"error": f"failed to read {path.name}: {exc}"},
            )
            return
        ctype = _CONTENT_TYPES.get(path.suffix.lower(), "application/octet-stream")
        self.send_response(int(status))
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # ----- structured logging --------------------------------------------- #
    def log_message(self, format: str, *args: object) -> None:  # noqa: ARG002
        """Log a one-line request summary to stderr.

        Overrides ``BaseHTTPRequestHandler.log_message``; the ``format``
        parameter is part of the upstream contract but we render our own line.
        """
        status = args[1] if len(args) > 1 else "?"
        sys.stderr.write(
            f"[bench dashboard] {self.command} {self.path} {status}\n"
        )
        sys.stderr.flush()

    # ----- routing -------------------------------------------------------- #
    def do_GET(self) -> None:
        """Route GET requests to the matching handler."""
        path = self.path.split("?", 1)[0]

        if path == "/__health__":
            self._handle_health()
            return
        if path == "/__rescan__":
            self._handle_rescan()
            return

        # Live data routes: rebuild if cache is stale, then serve from disk.
        self.server.maybe_rebuild()
        self._handle_static(path)

    # ----- handlers ------------------------------------------------------- #
    def _handle_health(self) -> None:
        self._send_json(
            HTTPStatus.OK,
            {
                "status": "ok",
                "envelopes": self.server.envelopes_count,
                "last_render_iso": self.server.last_render_iso,
            },
        )

    def _handle_rescan(self) -> None:
        self.server.force_rebuild()
        self._send_json(
            HTTPStatus.OK,
            {"rebuilt": True, "envelopes": self.server.envelopes_count},
        )

    def _handle_static(self, path: str) -> None:
        relative = "index.html" if path in ("", "/") else path.lstrip("/")
        # Reject path-traversal attempts up-front.
        if ".." in relative.split("/"):
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "invalid path"})
            return

        cache_dir = self.server.cache_dir
        candidate = (cache_dir / relative).resolve()
        try:
            candidate.relative_to(cache_dir.resolve())
        except ValueError:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "invalid path"})
            return

        # Directory request → serve its index.html if present.
        if candidate.is_dir():
            candidate = candidate / "index.html"

        if not candidate.is_file():
            self._send_json(
                HTTPStatus.NOT_FOUND, {"error": f"not found: {path}"}
            )
            return

        self._send_file(HTTPStatus.OK, candidate)
