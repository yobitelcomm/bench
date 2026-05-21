"""Tests for the SGLang engine adapter."""

from __future__ import annotations

import urllib.error
import urllib.request
from io import BytesIO
from pathlib import Path
from typing import Any

import pytest

from inferencebench_llm import EngineKind, LLMInferencePlugin, RunContext
from inferencebench_llm.engines import EngineUnavailableError, SGLangEngine
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
def test_sglang_engine_name() -> None:
    assert SGLangEngine.name == "sglang"
    assert SGLangEngine().name == "sglang"


def test_engine_registry_dispatches_sglang() -> None:
    engine = _engine_for(EngineKind.SGLANG)
    assert isinstance(engine, SGLangEngine)


def test_plugin_validate_accepts_sglang_engine_kind() -> None:
    """validate() should not flag SGLang as unimplemented now that it ships."""
    plugin = LLMInferencePlugin()
    spec = plugin.get_benchmark("llm.inference.sharegpt-v3")
    ctx = RunContext(
        model_id="meta-llama/Llama-4-Maverick",
        engine_kind=EngineKind.SGLANG,
        base_url="http://127.0.0.1:1/v1",  # unreachable on purpose
        output_dir=Path("/tmp/bench"),
    )
    warnings = plugin.validate(spec, ctx)
    # The engine probe will fail (unreachable) but the message must NOT be
    # the "not implemented" message — that would mean the registry missed it.
    assert not any("not implemented" in w.lower() for w in warnings)


# --------------------------------------------------------------------------- #
# probe()                                                                     #
# --------------------------------------------------------------------------- #
def test_sglang_probe_requires_base_url() -> None:
    engine = SGLangEngine()
    ctx = RunContext(
        model_id="m",
        engine_kind=EngineKind.SGLANG,
        base_url="",
        output_dir=Path("/tmp/bench"),
    )
    with pytest.raises(EngineUnavailableError, match="requires"):
        engine.probe(ctx)


def test_sglang_probe_unreachable_endpoint() -> None:
    engine = SGLangEngine()
    ctx = RunContext(
        model_id="m",
        engine_kind=EngineKind.SGLANG,
        base_url="http://127.0.0.1:1/v1",
        output_dir=Path("/tmp/bench"),
    )
    with pytest.raises(EngineUnavailableError, match="SGLang"):
        engine.probe(ctx)


def test_sglang_probe_reads_version_from_get_server_info(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = SGLangEngine()
    ctx = RunContext(
        model_id="m",
        engine_kind=EngineKind.SGLANG,
        base_url="http://localhost:30000/v1",
        output_dir=Path("/tmp/bench"),
    )

    captured: dict[str, str] = {}

    def fake_urlopen(req: urllib.request.Request, timeout: float = 5) -> _FakeResp:
        captured["url"] = req.full_url
        return _FakeResp(
            b'{"version": "0.4.5", "server_args": {"model_path": "Qwen/Qwen2.5-7B-Instruct"}}'
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    version = engine.probe(ctx)
    assert version == "0.4.5"
    assert captured["url"].endswith("/get_server_info")
    # Trailing /v1 should have been stripped for the info call.
    assert "/v1/get_server_info" not in captured["url"]


def test_sglang_probe_falls_back_to_v1_models_on_404(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = SGLangEngine()
    ctx = RunContext(
        model_id="m",
        engine_kind=EngineKind.SGLANG,
        base_url="http://localhost:30000",
        output_dir=Path("/tmp/bench"),
    )

    seen: list[str] = []

    def fake_urlopen(req: urllib.request.Request, timeout: float = 5) -> _FakeResp:
        url = req.full_url
        seen.append(url)
        if url.endswith("/get_server_info"):
            raise urllib.error.HTTPError(
                url=url,
                code=404,
                msg="Not Found",
                hdrs=None,
                fp=None,  # type: ignore[arg-type]
            )
        if url.endswith("/v1/models"):
            return _FakeResp(b'{"data": [{"id": "Qwen/Qwen2.5-7B-Instruct"}]}')
        raise AssertionError(f"unexpected url: {url}")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    version = engine.probe(ctx)
    assert version == "unknown"
    assert any(u.endswith("/get_server_info") for u in seen)
    assert any(u.endswith("/v1/models") for u in seen)


# --------------------------------------------------------------------------- #
# build_client()                                                              #
# --------------------------------------------------------------------------- #
def test_sglang_build_client_strips_openai_prefix() -> None:
    engine = SGLangEngine()
    ctx = RunContext(
        model_id="openai/Qwen/Qwen2.5-7B-Instruct",
        engine_kind=EngineKind.SGLANG,
        base_url="http://localhost:30000/v1",
        output_dir=Path("/tmp/bench"),
    )
    client = engine.build_client(ctx)
    # Exactly one ``openai/`` prefix, no doubling.
    assert client.model == "openai/Qwen/Qwen2.5-7B-Instruct"
    assert not client.model.startswith("openai/openai/")


def test_sglang_build_client_normalises_base_url() -> None:
    engine = SGLangEngine()
    for url in ("http://localhost:30000", "http://localhost:30000/", "http://localhost:30000/v1"):
        ctx = RunContext(
            model_id="m",
            engine_kind=EngineKind.SGLANG,
            base_url=url,
            output_dir=Path("/tmp/bench"),
        )
        client = engine.build_client(ctx)
        assert client.base_url == "http://localhost:30000/v1"
