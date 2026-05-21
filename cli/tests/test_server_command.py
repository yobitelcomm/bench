"""Tests for ``bench server``."""

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

import pytest
from _helpers import (  # type: ignore[import-not-found]
    make_envelope,
    write_envelope_json,
    write_signed_envelope_json,
)

from inferencebench.commands.server import _EnvelopeServer, make_server
from inferencebench.envelope import Envelope


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
@contextmanager
def _running_server(
    store: Path, dev_public_key: Path | None
) -> Iterator[tuple[_EnvelopeServer, str]]:
    """Start ``make_server`` on an ephemeral port in a worker thread; yield base URL."""
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

    # Tiny health-check spin until the listener is accepting connections.
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
        raise RuntimeError("server failed to start in 5s")

    try:
        yield httpd, base_url
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=2.0)


def _http_get(url: str) -> tuple[int, dict[str, object] | bytes]:
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            body = resp.read()
            status = resp.status
    except urllib.error.HTTPError as exc:
        body = exc.read()
        status = exc.code
    try:
        return status, json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return status, body


def _http_post(url: str, payload: bytes) -> tuple[int, dict[str, object] | bytes]:
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            body = resp.read()
            status = resp.status
    except urllib.error.HTTPError as exc:
        body = exc.read()
        status = exc.code
    try:
        return status, json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return status, body


def _signed_envelope_bytes(
    dev_key: Path, *, model_id: str = "m", run_suffix: str = "001"
) -> tuple[Envelope, bytes]:
    """Build + sign an envelope, return (envelope, canonical JSON bytes)."""
    from inferencebench.envelope import SigningMode, sign_envelope

    envelope = make_envelope(
        model_id=model_id,
        run_id=f"01934567-89ab-7000-8000-000000000{run_suffix}",
        metrics={"throughput_tok_per_s": 1234.5},
    )
    signed = sign_envelope(envelope, mode=SigningMode.DEV, dev_key_path=dev_key)
    body = json.dumps(signed.model_dump(mode="json"), sort_keys=True).encode("utf-8")
    return signed, body


# --------------------------------------------------------------------------- #
# Tests                                                                       #
# --------------------------------------------------------------------------- #
def test_server_health(tmp_path: Path) -> None:
    with _running_server(tmp_path / "store", None) as (_, base_url):
        status, body = _http_get(f"{base_url}/health")
    assert status == 200
    assert isinstance(body, dict)
    assert body["status"] == "ok"
    assert "version" in body


def test_server_post_signed_envelope_lands_on_disk(
    tmp_path: Path, dev_keypair: tuple[Path, Path]
) -> None:
    priv, pub = dev_keypair
    store = tmp_path / "store"
    with _running_server(store, pub) as (_, base_url):
        envelope, body = _signed_envelope_bytes(priv)
        status, resp = _http_post(f"{base_url}/envelopes", body)
    assert status == 201, resp
    assert isinstance(resp, dict)
    assert resp["content_hash"] == envelope.content_hash()
    expected_path = store / f"{envelope.content_hash()[:12]}.json"
    assert expected_path.exists()
    # The on-disk envelope must round-trip back through the schema.
    Envelope.model_validate(json.loads(expected_path.read_text(encoding="utf-8")))


def test_server_post_unsigned_envelope_rejected(tmp_path: Path) -> None:
    store = tmp_path / "store"
    envelope = make_envelope(model_id="m", metrics={"throughput_tok_per_s": 1.0})
    unsigned_path = write_envelope_json(tmp_path / "u.json", envelope)
    body = unsigned_path.read_bytes()
    with _running_server(store, None) as (_, base_url):
        status, resp = _http_post(f"{base_url}/envelopes", body)
    assert status == 401, resp


def test_server_post_tampered_envelope_rejected(
    tmp_path: Path, dev_keypair: tuple[Path, Path]
) -> None:
    priv, pub = dev_keypair
    store = tmp_path / "store"
    envelope = make_envelope(model_id="m", metrics={"throughput_tok_per_s": 1.0})
    signed_path = write_signed_envelope_json(tmp_path / "s.json", envelope, dev_key=priv)
    raw = json.loads(signed_path.read_text(encoding="utf-8"))
    raw["metrics"]["throughput_tok_per_s"] = 9999.0
    tampered = json.dumps(raw).encode("utf-8")
    with _running_server(store, pub) as (_, base_url):
        status, resp = _http_post(f"{base_url}/envelopes", tampered)
    assert status == 401, resp


def test_server_list_envelopes_returns_entries(
    tmp_path: Path, dev_keypair: tuple[Path, Path]
) -> None:
    priv, pub = dev_keypair
    store = tmp_path / "store"
    with _running_server(store, pub) as (_, base_url):
        _, b1 = _signed_envelope_bytes(priv, model_id="alpha", run_suffix="011")
        _, b2 = _signed_envelope_bytes(priv, model_id="beta", run_suffix="012")
        assert _http_post(f"{base_url}/envelopes", b1)[0] == 201
        assert _http_post(f"{base_url}/envelopes", b2)[0] == 201
        status, body = _http_get(f"{base_url}/envelopes")
    assert status == 200, body
    assert isinstance(body, dict)
    entries = body["entries"]
    assert isinstance(entries, list)
    assert len(entries) == 2
    model_ids = {entry["model_id"] for entry in entries}
    assert model_ids == {"alpha", "beta"}


def test_server_get_envelope_by_hash(tmp_path: Path, dev_keypair: tuple[Path, Path]) -> None:
    priv, pub = dev_keypair
    store = tmp_path / "store"
    with _running_server(store, pub) as (_, base_url):
        envelope, body = _signed_envelope_bytes(priv)
        assert _http_post(f"{base_url}/envelopes", body)[0] == 201
        ch = envelope.content_hash()
        status, fetched = _http_get(f"{base_url}/envelopes/{ch}")
        assert status == 200
        assert isinstance(fetched, dict)
        assert fetched.get("run_id") == envelope.run_id

        # Unknown hash → 404
        status404, _ = _http_get(f"{base_url}/envelopes/{'0' * 12}")
        assert status404 == 404


def test_server_invalid_json_body_returns_400(
    tmp_path: Path, dev_keypair: tuple[Path, Path]
) -> None:
    _, pub = dev_keypair
    store = tmp_path / "store"
    with _running_server(store, pub) as (_, base_url):
        status, resp = _http_post(f"{base_url}/envelopes", b"{not valid json}")
    assert status == 400, resp


@pytest.mark.parametrize(
    "method_path,expected",
    [
        ("/envelopes/", 400),
        ("/unknown", 404),
    ],
)
def test_server_unexpected_paths(tmp_path: Path, method_path: str, expected: int) -> None:
    with _running_server(tmp_path / "store", None) as (_, base_url):
        status, _ = _http_get(f"{base_url}{method_path}")
    assert status == expected
