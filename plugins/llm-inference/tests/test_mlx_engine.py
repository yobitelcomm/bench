"""Tests for the MLX engine adapter (``mlx_lm.server``)."""

from __future__ import annotations

import urllib.error
import urllib.request
from io import BytesIO
from pathlib import Path
from typing import Any

import pytest

from inferencebench_llm import EngineKind, RunContext
from inferencebench_llm.engines import EngineUnavailableError, MLXEngine
from inferencebench_llm.plugin import _engine_for


class _FakeResp:
    """Minimal context-manager response stand-in for ``urllib.request.urlopen``."""

    def __init__(self, body: bytes, headers: dict[str, str] | None = None) -> None:
        self._buf = BytesIO(body)
        self.headers: dict[str, str] = dict(headers) if headers else {}

    def __enter__(self) -> _FakeResp:
        return self

    def __exit__(self, *exc: Any) -> None:
        self._buf.close()

    def read(self) -> bytes:
        return self._buf.read()


# --------------------------------------------------------------------------- #
# Engine identity + registry wiring                                           #
# --------------------------------------------------------------------------- #
def test_mlx_engine_name() -> None:
    assert MLXEngine.name == "mlx"
    assert MLXEngine().name == "mlx"


def test_engine_registry_dispatches_mlx() -> None:
    engine = _engine_for(EngineKind.MLX)
    assert isinstance(engine, MLXEngine)


# --------------------------------------------------------------------------- #
# probe()                                                                     #
# --------------------------------------------------------------------------- #
def test_mlx_probe_requires_base_url() -> None:
    engine = MLXEngine()
    ctx = RunContext(
        model_id="m",
        engine_kind=EngineKind.MLX,
        base_url="",
        output_dir=Path("/tmp/bench"),
    )
    with pytest.raises(EngineUnavailableError, match="requires"):
        engine.probe(ctx)


def test_mlx_probe_unreachable_endpoint() -> None:
    """Connection failure must surface the mlx_lm.server launch hint."""
    engine = MLXEngine()
    ctx = RunContext(
        model_id="m",
        engine_kind=EngineKind.MLX,
        base_url="http://127.0.0.1:1/v1",
        output_dir=Path("/tmp/bench"),
    )
    with pytest.raises(EngineUnavailableError, match=r"mlx_lm\.server"):
        engine.probe(ctx)


def test_mlx_probe_happy_path_returns_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``/v1/models`` 200 -> ``"unknown"`` (mlx_lm.server doesn't expose a version)."""
    engine = MLXEngine()
    ctx = RunContext(
        model_id="m",
        engine_kind=EngineKind.MLX,
        base_url="http://localhost:8000/v1",
        output_dir=Path("/tmp/bench"),
    )

    captured: dict[str, str] = {}

    def fake_urlopen(req: urllib.request.Request, timeout: float = 5) -> _FakeResp:
        captured["url"] = req.full_url
        return _FakeResp(
            b'{"data": [{"id": "mlx-community/Llama-3.1-8B-Instruct-4bit"}]}',
            headers={},
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    version = engine.probe(ctx)
    assert version == "unknown"
    # The trailing /v1 in the supplied base_url must NOT have been doubled.
    assert captured["url"].endswith("/v1/models")
    assert "/v1/v1/models" not in captured["url"]


# --------------------------------------------------------------------------- #
# build_client()                                                              #
# --------------------------------------------------------------------------- #
def test_mlx_build_client_strips_openai_prefix() -> None:
    engine = MLXEngine()
    ctx = RunContext(
        model_id="openai/mlx-community/Llama-3.1-8B-Instruct-4bit",
        engine_kind=EngineKind.MLX,
        base_url="http://localhost:8000/v1",
        output_dir=Path("/tmp/bench"),
    )
    client = engine.build_client(ctx)
    # Exactly one ``openai/`` prefix, no doubling.
    assert client.model == "openai/mlx-community/Llama-3.1-8B-Instruct-4bit"
    assert not client.model.startswith("openai/openai/")


def test_mlx_build_client_normalises_base_url() -> None:
    """base_url variants all collapse to a canonical .../v1."""
    engine = MLXEngine()
    for url in ("http://localhost:8000", "http://localhost:8000/", "http://localhost:8000/v1"):
        ctx = RunContext(
            model_id="m",
            engine_kind=EngineKind.MLX,
            base_url=url,
            output_dir=Path("/tmp/bench"),
        )
        client = engine.build_client(ctx)
        assert client.base_url == "http://localhost:8000/v1"


def test_mlx_build_client_uses_empty_api_key_by_default() -> None:
    """No api_key on the context -> LiteLLM gets the sentinel ``"EMPTY"``."""
    engine = MLXEngine()
    ctx = RunContext(
        model_id="m",
        engine_kind=EngineKind.MLX,
        base_url="http://localhost:8000/v1",
        output_dir=Path("/tmp/bench"),
    )
    client = engine.build_client(ctx)
    assert client.api_key == "EMPTY"


def test_mlx_probe_timeout_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """A socket timeout must surface as ``EngineUnavailableError`` with the launch hint."""
    engine = MLXEngine()
    ctx = RunContext(
        model_id="m",
        engine_kind=EngineKind.MLX,
        base_url="http://localhost:8000/v1",
        output_dir=Path("/tmp/bench"),
    )

    def fake_urlopen(req: urllib.request.Request, timeout: float = 5) -> _FakeResp:
        raise urllib.error.URLError("timed out")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(EngineUnavailableError, match=r"mlx_lm\.server"):
        engine.probe(ctx)
