"""Tests for the llama.cpp engine adapter."""

from __future__ import annotations

import urllib.error
import urllib.request
from io import BytesIO
from pathlib import Path
from typing import Any

import pytest

from inferencebench_llm import EngineKind, LLMInferencePlugin, RunContext
from inferencebench_llm.engines import EngineUnavailableError, LlamaCppEngine
from inferencebench_llm.plugin import _engine_for


class _FakeResp:
    """Minimal context-manager response stand-in for ``urllib.request.urlopen``."""

    def __init__(self, body: bytes) -> None:
        self._buf = BytesIO(body)

    def __enter__(self) -> _FakeResp:
        return self

    def __exit__(self, *exc: Any) -> None:
        self._buf.close()

    def read(self) -> bytes:
        return self._buf.read()


# --------------------------------------------------------------------------- #
# Engine identity + registry wiring                                           #
# --------------------------------------------------------------------------- #
def test_llamacpp_engine_name() -> None:
    assert LlamaCppEngine.name == "llamacpp"
    assert LlamaCppEngine().name == "llamacpp"


def test_engine_registry_dispatches_llamacpp() -> None:
    engine = _engine_for(EngineKind.LLAMACPP)
    assert isinstance(engine, LlamaCppEngine)


def test_plugin_validate_accepts_llamacpp_engine_kind() -> None:
    """validate() should not flag llama.cpp as unimplemented now that it ships."""
    plugin = LLMInferencePlugin()
    spec = plugin.get_benchmark("llm.inference.sharegpt-v3")
    ctx = RunContext(
        model_id="meta-llama/Llama-3.1-8B-Instruct",
        engine_kind=EngineKind.LLAMACPP,
        base_url="http://127.0.0.1:1/v1",  # unreachable on purpose
        output_dir=Path("/tmp/bench"),
    )
    warnings = plugin.validate(spec, ctx)
    assert not any("not implemented" in w.lower() for w in warnings)


# --------------------------------------------------------------------------- #
# probe()                                                                     #
# --------------------------------------------------------------------------- #
def test_llamacpp_probe_requires_base_url() -> None:
    engine = LlamaCppEngine()
    ctx = RunContext(
        model_id="m",
        engine_kind=EngineKind.LLAMACPP,
        base_url="",
        output_dir=Path("/tmp/bench"),
    )
    with pytest.raises(EngineUnavailableError, match="requires"):
        engine.probe(ctx)


def test_llamacpp_probe_unreachable_endpoint() -> None:
    engine = LlamaCppEngine()
    ctx = RunContext(
        model_id="m",
        engine_kind=EngineKind.LLAMACPP,
        base_url="http://127.0.0.1:1/v1",
        output_dir=Path("/tmp/bench"),
    )
    with pytest.raises(EngineUnavailableError, match="llama"):
        engine.probe(ctx)


def test_llamacpp_probe_reads_system_info_from_props(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = LlamaCppEngine()
    ctx = RunContext(
        model_id="m",
        engine_kind=EngineKind.LLAMACPP,
        base_url="http://localhost:8000/v1",
        output_dir=Path("/tmp/bench"),
    )

    captured: dict[str, str] = {}

    def fake_urlopen(req: urllib.request.Request, timeout: float = 5) -> _FakeResp:
        captured["url"] = req.full_url
        return _FakeResp(
            b'{"system_info": "AVX2 1 | AVX_VNNI 0 | AVX512 1 | ... llama.cpp build 1.2.3", '
            b'"model_path": "/tmp/m.gguf", "n_ctx": 4096}'
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    version = engine.probe(ctx)
    assert "AVX2" in version
    assert captured["url"].endswith("/props")
    # Trailing /v1 should have been stripped for the props call.
    assert "/v1/props" not in captured["url"]


def test_llamacpp_probe_falls_back_to_v1_models_on_404(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = LlamaCppEngine()
    ctx = RunContext(
        model_id="m",
        engine_kind=EngineKind.LLAMACPP,
        base_url="http://localhost:8000",
        output_dir=Path("/tmp/bench"),
    )

    seen: list[str] = []

    def fake_urlopen(req: urllib.request.Request, timeout: float = 5) -> _FakeResp:
        url = req.full_url
        seen.append(url)
        if url.endswith("/props"):
            raise urllib.error.HTTPError(
                url=url, code=404, msg="Not Found", hdrs=None, fp=None  # type: ignore[arg-type]
            )
        if url.endswith("/v1/models"):
            return _FakeResp(b'{"data": [{"id": "llama-3.1-8b-q4_k_m.gguf"}]}')
        raise AssertionError(f"unexpected url: {url}")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    version = engine.probe(ctx)
    assert version == "unknown"
    assert any(u.endswith("/props") for u in seen)
    assert any(u.endswith("/v1/models") for u in seen)


# --------------------------------------------------------------------------- #
# build_client()                                                              #
# --------------------------------------------------------------------------- #
def test_llamacpp_build_client_strips_openai_prefix() -> None:
    engine = LlamaCppEngine()
    ctx = RunContext(
        model_id="openai/llama-3.1-8b-q4_k_m.gguf",
        engine_kind=EngineKind.LLAMACPP,
        base_url="http://localhost:8000/v1",
        output_dir=Path("/tmp/bench"),
    )
    client = engine.build_client(ctx)
    # Exactly one ``openai/`` prefix, no doubling.
    assert client.model == "openai/llama-3.1-8b-q4_k_m.gguf"
    assert not client.model.startswith("openai/openai/")


def test_llamacpp_build_client_normalises_base_url() -> None:
    engine = LlamaCppEngine()
    for url in ("http://localhost:8000", "http://localhost:8000/", "http://localhost:8000/v1"):
        ctx = RunContext(
            model_id="m",
            engine_kind=EngineKind.LLAMACPP,
            base_url=url,
            output_dir=Path("/tmp/bench"),
        )
        client = engine.build_client(ctx)
        assert client.base_url == "http://localhost:8000/v1"
