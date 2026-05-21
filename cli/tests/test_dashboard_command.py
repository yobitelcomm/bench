"""Tests for ``bench dashboard`` — live HTTP leaderboard server."""

from __future__ import annotations

import json
import socket
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from _helpers import make_envelope, write_envelope_json  # type: ignore[import-not-found]

from inferencebench.commands.dashboard import _DashboardServer, make_dashboard_server


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
@contextmanager
def _running_dashboard(
    envelopes_dir: Path, *, rebuild_interval_s: float = 0.0
) -> Iterator[tuple[_DashboardServer, str]]:
    """Start ``make_dashboard_server`` on an ephemeral port; yield server + base URL."""
    httpd = make_dashboard_server(
        host="127.0.0.1",
        port=0,
        envelopes_dir=envelopes_dir,
        rebuild_interval_s=rebuild_interval_s,
    )
    host, port = httpd.server_address[:2]
    base_url = f"http://{host}:{port}"

    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()

    deadline = time.time() + 5.0
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.1):
                break
        except OSError:
            time.sleep(0.02)
    else:  # pragma: no cover - shouldn't happen on a healthy box
        httpd.shutdown()
        httpd.server_close()
        httpd.cleanup()
        raise RuntimeError("dashboard failed to start in 5s")

    try:
        yield httpd, base_url
    finally:
        httpd.shutdown()
        httpd.server_close()
        httpd.cleanup()
        thread.join(timeout=2.0)


def _http_get(url: str) -> tuple[int, bytes, str]:
    """GET ``url`` and return (status, body bytes, content-type)."""
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            body = resp.read()
            status = resp.status
            ctype = resp.headers.get("Content-Type", "")
    except urllib.error.HTTPError as exc:
        body = exc.read()
        status = exc.code
        ctype = exc.headers.get("Content-Type", "") if exc.headers else ""
    return status, body, ctype


def _http_get_json(url: str) -> tuple[int, dict[str, Any]]:
    status, body, _ = _http_get(url)
    return status, json.loads(body.decode("utf-8"))


def _write_envelope(envelopes_dir: Path, *, model_id: str, run_suffix: str) -> Path:
    return write_envelope_json(
        envelopes_dir / f"{run_suffix}.json",
        make_envelope(
            model_id=model_id,
            run_id=f"01934567-89ab-7000-8000-{run_suffix:>012}",
            metrics={
                "throughput_tok_per_s": 1500.0,
                "ttft_p99_ms": 400.0,
                "cost_usd_per_million_tokens": 0.5,
            },
        ),
    )


# --------------------------------------------------------------------------- #
# Tests                                                                       #
# --------------------------------------------------------------------------- #
def test_dashboard_health_endpoint(tmp_path: Path) -> None:
    """GET /__health__ → 200 JSON with status=ok and envelopes count."""
    envelopes_dir = tmp_path / "envelopes"
    envelopes_dir.mkdir()
    _write_envelope(envelopes_dir, model_id="alpha", run_suffix="001")

    with _running_dashboard(envelopes_dir) as (_, base_url):
        status, body = _http_get_json(f"{base_url}/__health__")

    assert status == 200, body
    assert body["status"] == "ok"
    assert body["envelopes"] == 1
    assert "last_render_iso" in body


def test_dashboard_index_lists_both_models(tmp_path: Path) -> None:
    """With two envelopes, GET / returns HTML referencing both model ids."""
    envelopes_dir = tmp_path / "envelopes"
    envelopes_dir.mkdir()
    _write_envelope(envelopes_dir, model_id="meta-llama/Llama-4-Maverick", run_suffix="001")
    _write_envelope(envelopes_dir, model_id="mistralai/Mistral-Large", run_suffix="002")

    with _running_dashboard(envelopes_dir) as (_, base_url):
        status, body, ctype = _http_get(f"{base_url}/")
        assert status == 200
        assert "text/html" in ctype
        text = body.decode("utf-8")
        # Top-level index lists categories; per-suite page lists model ids.
        # Verify the per-suite page renders both models.
        suite_status, suite_body, _ = _http_get(f"{base_url}/llm.inference/")

    assert suite_status == 200, suite_body
    suite_text = suite_body.decode("utf-8")
    assert "meta-llama/Llama-4-Maverick" in suite_text
    assert "mistralai/Mistral-Large" in suite_text
    # Sanity: the top-level index references the suite.
    assert "llm.inference" in text


def test_dashboard_rescan_picks_up_new_envelope(tmp_path: Path) -> None:
    """Writing a third envelope, then GET /__rescan__, updates the served view."""
    envelopes_dir = tmp_path / "envelopes"
    envelopes_dir.mkdir()
    _write_envelope(envelopes_dir, model_id="alpha", run_suffix="001")
    _write_envelope(envelopes_dir, model_id="beta", run_suffix="002")

    # Long rebuild interval so a normal GET / does NOT pick the new one up;
    # only explicit /__rescan__ should refresh.
    with _running_dashboard(envelopes_dir, rebuild_interval_s=3600.0) as (_, base_url):
        status, body = _http_get_json(f"{base_url}/__health__")
        assert status == 200
        assert body["envelopes"] == 2

        _write_envelope(envelopes_dir, model_id="gamma", run_suffix="003")

        rescan_status, rescan_body = _http_get_json(f"{base_url}/__rescan__")
        assert rescan_status == 200, rescan_body
        assert rescan_body["rebuilt"] is True
        assert rescan_body["envelopes"] == 3

        # Subsequent GET / reflects the third envelope.
        suite_status, suite_body, _ = _http_get(f"{base_url}/llm.inference/")
        assert suite_status == 200
        assert "gamma" in suite_body.decode("utf-8")


def test_dashboard_data_leaderboard_json_parseable(tmp_path: Path) -> None:
    """GET /data/leaderboard.json returns parseable JSON with the expected shape."""
    envelopes_dir = tmp_path / "envelopes"
    envelopes_dir.mkdir()
    _write_envelope(envelopes_dir, model_id="alpha", run_suffix="001")

    with _running_dashboard(envelopes_dir) as (_, base_url):
        status, body, ctype = _http_get(f"{base_url}/data/leaderboard.json")

    assert status == 200
    assert "application/json" in ctype
    payload = json.loads(body.decode("utf-8"))
    assert "categories" in payload
    total = sum(len(c["entries"]) for c in payload["categories"])
    assert total == 1


def test_dashboard_nonexistent_path_returns_404(tmp_path: Path) -> None:
    """GET /nonexistent → 404."""
    envelopes_dir = tmp_path / "envelopes"
    envelopes_dir.mkdir()
    _write_envelope(envelopes_dir, model_id="alpha", run_suffix="001")

    with _running_dashboard(envelopes_dir) as (_, base_url):
        status, _, _ = _http_get(f"{base_url}/nonexistent")

    assert status == 404
