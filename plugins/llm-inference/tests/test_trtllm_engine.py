"""Tests for the TensorRT-LLM engine adapter."""

from __future__ import annotations

import urllib.error
import urllib.request
from io import BytesIO
from pathlib import Path
from typing import Any

import pytest

from inferencebench_llm import EngineKind, LLMInferencePlugin, RunContext
from inferencebench_llm.engines import EngineUnavailableError, TRTLLMEngine
from inferencebench_llm.plugin import _engine_for


class _FakeResp:
    """Minimal context-manager response stand-in for ``urllib.request.urlopen``.

    Supports ``read()`` plus ``.headers.get(...)``, which is how the TRT-LLM
    adapter extracts the ``Server`` version header.
    """

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
def test_trtllm_engine_name() -> None:
    assert TRTLLMEngine.name == "trtllm"
    assert TRTLLMEngine().name == "trtllm"


def test_engine_registry_dispatches_trtllm() -> None:
    engine = _engine_for(EngineKind.TRTLLM)
    assert isinstance(engine, TRTLLMEngine)


def test_plugin_validate_accepts_trtllm_engine_kind() -> None:
    """validate() should not flag TRT-LLM as unimplemented now that it ships."""
    plugin = LLMInferencePlugin()
    spec = plugin.get_benchmark("llm.inference.sharegpt-v3")
    ctx = RunContext(
        model_id="meta-llama/Llama-3.1-8B-Instruct",
        engine_kind=EngineKind.TRTLLM,
        base_url="http://127.0.0.1:1/v1",  # unreachable on purpose
        output_dir=Path("/tmp/bench"),
    )
    warnings = plugin.validate(spec, ctx)
    # Probe will fail (unreachable) but message must NOT be the
    # "not implemented" message — that would mean the registry missed it.
    assert not any("not implemented" in w.lower() for w in warnings)


# --------------------------------------------------------------------------- #
# probe()                                                                     #
# --------------------------------------------------------------------------- #
def test_trtllm_probe_requires_base_url() -> None:
    engine = TRTLLMEngine()
    ctx = RunContext(
        model_id="m",
        engine_kind=EngineKind.TRTLLM,
        base_url="",
        output_dir=Path("/tmp/bench"),
    )
    with pytest.raises(EngineUnavailableError, match="requires"):
        engine.probe(ctx)


def test_trtllm_probe_unreachable_endpoint() -> None:
    engine = TRTLLMEngine()
    ctx = RunContext(
        model_id="m",
        engine_kind=EngineKind.TRTLLM,
        base_url="http://127.0.0.1:1/v1",
        output_dir=Path("/tmp/bench"),
    )
    with pytest.raises(EngineUnavailableError, match="TRT-LLM"):
        engine.probe(ctx)


def test_trtllm_probe_reads_version_from_server_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``/health/load`` returns 200 with ``Server: trtllm-0.13.0`` -> ``0.13.0``."""
    engine = TRTLLMEngine()
    ctx = RunContext(
        model_id="m",
        engine_kind=EngineKind.TRTLLM,
        base_url="http://localhost:8000/v1",
        output_dir=Path("/tmp/bench"),
    )

    captured: dict[str, str] = {}

    def fake_urlopen(req: urllib.request.Request, timeout: float = 5) -> _FakeResp:
        captured["url"] = req.full_url
        return _FakeResp(
            b'{"queue_length": 0, "kv_cache_used_ratio": 0.0}',
            headers={"Server": "trtllm-0.13.0"},
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    version = engine.probe(ctx)
    assert version == "0.13.0"
    assert captured["url"].endswith("/health/load")
    # Trailing /v1 should have been stripped for the health call.
    assert "/v1/health/load" not in captured["url"]


def test_trtllm_probe_falls_back_to_models_when_header_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """200 from ``/health/load`` with no ``Server`` header -> try ``/v1/models`` -> ``unknown``."""
    engine = TRTLLMEngine()
    ctx = RunContext(
        model_id="m",
        engine_kind=EngineKind.TRTLLM,
        base_url="http://localhost:8000",
        output_dir=Path("/tmp/bench"),
    )

    seen: list[str] = []

    def fake_urlopen(req: urllib.request.Request, timeout: float = 5) -> _FakeResp:
        url = req.full_url
        seen.append(url)
        if url.endswith("/health/load"):
            return _FakeResp(
                b'{"queue_length": 0, "kv_cache_used_ratio": 0.0}',
                headers={},  # no Server header
            )
        if url.endswith("/v1/models"):
            return _FakeResp(
                b'{"data": [{"id": "meta-llama/Llama-3.1-8B-Instruct"}]}',
                headers={},  # no Server header here either
            )
        raise AssertionError(f"unexpected url: {url}")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    version = engine.probe(ctx)
    assert version == "unknown"
    assert any(u.endswith("/health/load") for u in seen)
    assert any(u.endswith("/v1/models") for u in seen)


def test_trtllm_probe_falls_back_to_models_on_404(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``/health/load`` returns 404 -> fall back to ``/v1/models`` -> ``unknown``."""
    engine = TRTLLMEngine()
    ctx = RunContext(
        model_id="m",
        engine_kind=EngineKind.TRTLLM,
        base_url="http://localhost:8000",
        output_dir=Path("/tmp/bench"),
    )

    seen: list[str] = []

    def fake_urlopen(req: urllib.request.Request, timeout: float = 5) -> _FakeResp:
        url = req.full_url
        seen.append(url)
        if url.endswith("/health/load"):
            raise urllib.error.HTTPError(
                url=url, code=404, msg="Not Found", hdrs=None, fp=None  # type: ignore[arg-type]
            )
        if url.endswith("/v1/models"):
            return _FakeResp(
                b'{"data": [{"id": "meta-llama/Llama-3.1-8B-Instruct"}]}',
                headers={},
            )
        raise AssertionError(f"unexpected url: {url}")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    version = engine.probe(ctx)
    assert version == "unknown"
    assert any(u.endswith("/health/load") for u in seen)
    assert any(u.endswith("/v1/models") for u in seen)


# --------------------------------------------------------------------------- #
# build_client()                                                              #
# --------------------------------------------------------------------------- #
def test_trtllm_build_client_strips_openai_prefix() -> None:
    engine = TRTLLMEngine()
    ctx = RunContext(
        model_id="openai/meta-llama/Llama-3.1-8B-Instruct",
        engine_kind=EngineKind.TRTLLM,
        base_url="http://localhost:8000/v1",
        output_dir=Path("/tmp/bench"),
    )
    client = engine.build_client(ctx)
    # Exactly one ``openai/`` prefix, no doubling.
    assert client.model == "openai/meta-llama/Llama-3.1-8B-Instruct"
    assert not client.model.startswith("openai/openai/")


def test_trtllm_build_client_normalises_base_url() -> None:
    engine = TRTLLMEngine()
    for url in ("http://localhost:8000", "http://localhost:8000/", "http://localhost:8000/v1"):
        ctx = RunContext(
            model_id="m",
            engine_kind=EngineKind.TRTLLM,
            base_url=url,
            output_dir=Path("/tmp/bench"),
        )
        client = engine.build_client(ctx)
        assert client.base_url == "http://localhost:8000/v1"
