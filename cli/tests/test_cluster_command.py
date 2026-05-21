"""Tests for ``bench cluster`` — runner-side coordinator."""

from __future__ import annotations

import socket
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from _helpers import make_envelope  # type: ignore[import-not-found]
from typer.testing import CliRunner

from inferencebench.cli import app
from inferencebench.commands.server import _EnvelopeServer, make_server
from inferencebench.envelope import SigningMode, generate_dev_keypair, sign_envelope

if TYPE_CHECKING:
    import pytest

runner = CliRunner(env={"COLUMNS": "240"})


# --------------------------------------------------------------------------- #
# Helpers — real ``bench server`` on an ephemeral port                        #
# --------------------------------------------------------------------------- #
@contextmanager
def _running_server(
    store: Path, dev_public_key: Path | None
) -> Iterator[tuple[_EnvelopeServer, str]]:
    """Start the real ``bench server`` in a worker thread on an ephemeral port."""
    httpd = make_server(
        host="127.0.0.1",
        port=0,
        store=store,
        dev_public_key=dev_public_key,
    )
    host, port = httpd.server_address[:2]
    base_url = f"http://{host}:{port}"

    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()

    # Spin until the listener accepts connections so we don't race the test.
    deadline = time.time() + 5.0
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.1):
                break
        except OSError:
            time.sleep(0.02)
    else:  # pragma: no cover
        httpd.shutdown()
        httpd.server_close()
        raise RuntimeError("server failed to start in 5s")

    try:
        yield httpd, base_url
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=2.0)


# --------------------------------------------------------------------------- #
# Tiny 401-everything server for the auth-rejection test                      #
# --------------------------------------------------------------------------- #
class _AlwaysUnauthorizedHandler(BaseHTTPRequestHandler):
    """Stub handler that returns 401 on POST and an empty list on GET /envelopes."""

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or "0")
        if length > 0:
            self.rfile.read(length)
        body = b'{"error":"unauthorized"}'
        self.send_response(int(HTTPStatus.UNAUTHORIZED))
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        body = b'{"entries": []}'
        self.send_response(int(HTTPStatus.OK))
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:  # noqa: ARG002
        return


@contextmanager
def _running_stub_server() -> Iterator[str]:
    """Spin a 401-on-POST stub server for the failed-POST path test."""
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _AlwaysUnauthorizedHandler)
    host, port = httpd.server_address[:2]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    deadline = time.time() + 5.0
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.1):
                break
        except OSError:
            time.sleep(0.02)
    try:
        yield f"http://{host}:{port}"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=2.0)


# --------------------------------------------------------------------------- #
# Plugin run stub (mirrors test_matrix_command.py)                            #
# --------------------------------------------------------------------------- #
def _install_fake_run(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    from inferencebench_llm.plugin import LLMInferencePlugin

    def fake_run(self: Any, spec: Any, context: Any) -> Any:  # noqa: ARG001
        target_name = str(context.extra.get("target_name", "unknown"))
        point = int(context.extra.get("concurrency", 0))
        calls.append({"target": target_name, "point": point})
        salt = abs(hash((target_name, point))) % 10**10
        env = make_envelope(
            model_id=f"fake-{target_name}",
            run_id=f"01934567-89ab-7000-8000-0{salt:011d}"[:36],
            metrics={
                "throughput_tok_per_s": 1000.0 + point * 10,
                "ttft_p50_ms": 100.0,
                "ttft_p99_ms": 250.0,
                "tpot_p50_ms": 20.0,
                "total_p50_ms": 1500.0,
                "ok_rate": 1.0,
                "compliance_rate": 0.97,
            },
        )
        dev_key_path = context.extra.get("dev_key_path")
        if dev_key_path:
            return sign_envelope(env, mode=SigningMode.DEV, dev_key_path=Path(str(dev_key_path)))
        return env

    def fake_validate(self: Any, spec: Any, context: Any) -> list[str]:  # noqa: ARG001
        return []

    monkeypatch.setattr(LLMInferencePlugin, "run", fake_run)
    monkeypatch.setattr(LLMInferencePlugin, "validate", fake_validate)
    return calls


def _two_by_two_config(tmp_path: Path) -> Path:
    cfg_path = tmp_path / "matrix.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "schema": "inferencebench.matrix.v1",
                "suite_id": "llm.inference",
                "sweep": [1, 4],
                "targets": [
                    {
                        "name": "vllm-a",
                        "model": "fake/model-a",
                        "engine": "vllm",
                        "base_url": "http://localhost:8000/v1",
                        "extra": {"target_name": "vllm-a"},
                    },
                    {
                        "name": "vllm-b",
                        "model": "fake/model-b",
                        "engine": "vllm",
                        "base_url": "http://localhost:8001/v1",
                        "extra": {"target_name": "vllm-b"},
                    },
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return cfg_path


def _dev_keypair(tmp_path: Path) -> tuple[Path, Path]:
    return generate_dev_keypair(tmp_path / "cosign.key")


# --------------------------------------------------------------------------- #
# Tests                                                                       #
# --------------------------------------------------------------------------- #
def test_cluster_run_posts_all_envelopes_to_server(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    priv, pub = _dev_keypair(tmp_path)
    _install_fake_run(monkeypatch)
    cfg = _two_by_two_config(tmp_path)
    out_dir = tmp_path / "results"
    store = tmp_path / "store"

    with _running_server(store, pub) as (_, base_url):
        result = runner.invoke(
            app,
            [
                "cluster",
                "run",
                str(cfg),
                "--output",
                str(out_dir),
                "--signing-mode",
                "dev",
                "--dev-key",
                str(priv),
                "--server-url",
                base_url,
            ],
        )
        # Snapshot the server's view inside the context manager so the store
        # is still accessible.
        landed = sorted(p.name for p in store.glob("*.json"))

    combined = result.stdout + (result.stderr or "")
    assert result.exit_code == 0, combined
    files = sorted(p.name for p in out_dir.glob("*.json"))
    # 2 targets x 2 sweep points = 4 envelopes.
    assert len(files) == 4, files
    assert len(landed) == 4, landed
    assert "POST summary" in combined
    assert "4 succeeded" in combined


def test_cluster_status_lists_envelopes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    priv, pub = _dev_keypair(tmp_path)
    _install_fake_run(monkeypatch)
    cfg = _two_by_two_config(tmp_path)
    out_dir = tmp_path / "results"
    store = tmp_path / "store"

    with _running_server(store, pub) as (_, base_url):
        run_result = runner.invoke(
            app,
            [
                "cluster",
                "run",
                str(cfg),
                "--output",
                str(out_dir),
                "--signing-mode",
                "dev",
                "--dev-key",
                str(priv),
                "--server-url",
                base_url,
            ],
        )
        assert run_result.exit_code == 0, run_result.stdout + (run_result.stderr or "")
        status_result = runner.invoke(app, ["cluster", "status", "--server-url", base_url])

    combined = status_result.stdout + (status_result.stderr or "")
    assert status_result.exit_code == 0, combined
    assert "4 envelope(s)" in combined
    assert "fake-vllm-a" in combined or "fake-vllm-b" in combined


def test_cluster_sync_writes_envelopes_to_disk(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    priv, pub = _dev_keypair(tmp_path)
    _install_fake_run(monkeypatch)
    cfg = _two_by_two_config(tmp_path)
    out_dir = tmp_path / "results"
    store = tmp_path / "store"
    sync_dir = tmp_path / "synced"

    with _running_server(store, pub) as (_, base_url):
        run_result = runner.invoke(
            app,
            [
                "cluster",
                "run",
                str(cfg),
                "--output",
                str(out_dir),
                "--signing-mode",
                "dev",
                "--dev-key",
                str(priv),
                "--server-url",
                base_url,
            ],
        )
        assert run_result.exit_code == 0
        sync_result = runner.invoke(
            app,
            [
                "cluster",
                "sync",
                "--server-url",
                base_url,
                "--out",
                str(sync_dir),
            ],
        )
        combined = sync_result.stdout + (sync_result.stderr or "")
        assert sync_result.exit_code == 0, combined
        synced_files = sorted(p.name for p in sync_dir.glob("*.json"))
        assert len(synced_files) == 4, synced_files

        # A second sync is idempotent — all 4 should be reported as skipped.
        again = runner.invoke(
            app,
            [
                "cluster",
                "sync",
                "--server-url",
                base_url,
                "--out",
                str(sync_dir),
            ],
        )
    again_combined = again.stdout + (again.stderr or "")
    assert again.exit_code == 0, again_combined
    files_after = sorted(p.name for p in sync_dir.glob("*.json"))
    assert len(files_after) == 4, files_after


def test_cluster_status_unreachable_server_exits_1(tmp_path: Path) -> None:
    # Bind a socket on an ephemeral port and immediately close it so the
    # follow-up connect attempt is refused (no listener) deterministically.
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    _, port = sock.getsockname()
    sock.close()
    bad_url = f"http://127.0.0.1:{port}"

    result = runner.invoke(app, ["cluster", "status", "--server-url", bad_url])
    combined = result.stdout + (result.stderr or "")
    assert result.exit_code == 1, combined
    assert "Cannot reach" in combined or "refused" in combined.lower()
    _ = tmp_path  # quiet the unused fixture warning


def test_cluster_run_continues_when_post_returns_401(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    priv, _pub = _dev_keypair(tmp_path)
    _install_fake_run(monkeypatch)
    cfg = _two_by_two_config(tmp_path)
    out_dir = tmp_path / "results"

    with _running_stub_server() as base_url:
        result = runner.invoke(
            app,
            [
                "cluster",
                "run",
                str(cfg),
                "--output",
                str(out_dir),
                "--signing-mode",
                "dev",
                "--dev-key",
                str(priv),
                "--server-url",
                base_url,
            ],
        )

    combined = result.stdout + (result.stderr or "")
    # All envelopes still produced + on disk locally.
    assert result.exit_code == 0, combined
    files = sorted(p.name for p in out_dir.glob("*.json"))
    assert len(files) == 4, files
    # And the user is told the POSTs failed.
    assert "warning" in combined.lower()
    assert "401" in combined
    assert "0 succeeded, 4 failed" in combined
