"""Tests for ``bench fetch`` — remote envelope URI -> local cache."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import pytest
from _helpers import (  # type: ignore[import-not-found]
    make_envelope,
    write_signed_envelope_json,
)
from typer.testing import CliRunner

from inferencebench.cli import app
from inferencebench.commands import fetch as fetch_mod

runner = CliRunner(env={"COLUMNS": "240"})


# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #
@pytest.fixture
def signed_envelope_on_disk(tmp_path: Path, dev_keypair: tuple[Path, Path]) -> Path:
    """A real, dev-signed envelope persisted to ``tmp_path/source.json``."""
    priv, _ = dev_keypair
    env = make_envelope(
        model_id="meta-llama/Llama-4-Test",
        run_id="01934567-89ab-7000-8000-000000fe1c41",
        metrics={
            "throughput_tok_per_s": 1500.0,
            "ttft_p99_ms": 400.0,
        },
    )
    path = tmp_path / "source.json"
    return write_signed_envelope_json(path, env, dev_key=priv)


@pytest.fixture
def isolated_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect the default cache dir to ``tmp_path/home`` for the test."""
    fake_home = tmp_path / "home"
    fake_home.mkdir(parents=True, exist_ok=True)

    def _fake_home() -> Path:
        return fake_home

    monkeypatch.setattr(Path, "home", staticmethod(_fake_home))
    return fake_home / ".cache" / "inferencebench" / "fetched"


class _FakeUrlResp:
    """Minimal ``urlopen`` context-manager stand-in returning a fixed payload."""

    def __init__(self, data: bytes) -> None:
        self._buf = io.BytesIO(data)

    def __enter__(self) -> _FakeUrlResp:
        return self

    def __exit__(self, *args: object) -> None:
        self._buf.close()

    def read(self) -> bytes:
        return self._buf.read()


# --------------------------------------------------------------------------- #
# Tests                                                                       #
# --------------------------------------------------------------------------- #
def test_fetch_file_scheme_copies_locally(
    signed_envelope_on_disk: Path,
    isolated_cache: Path,
) -> None:
    uri = f"file://{signed_envelope_on_disk}"
    result = runner.invoke(app, ["fetch", uri])
    assert result.exit_code == 0, result.output
    cached = list(isolated_cache.glob("*.json"))
    assert len(cached) == 1
    assert json.loads(cached[0].read_text())["suite_id"] == "llm.inference"
    assert "content_hash" in result.output
    assert "model_id" in result.output


def test_fetch_plain_path_works(
    signed_envelope_on_disk: Path,
    isolated_cache: Path,
) -> None:
    """No-scheme inputs are treated as local paths."""
    result = runner.invoke(app, ["fetch", str(signed_envelope_on_disk)])
    assert result.exit_code == 0, result.output
    cached = list(isolated_cache.glob("*.json"))
    assert len(cached) == 1


def test_fetch_https_uses_urlopen_monkeypatch(
    signed_envelope_on_disk: Path,
    isolated_cache: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = signed_envelope_on_disk.read_bytes()

    def fake_urlopen(url: str) -> _FakeUrlResp:
        assert url.startswith("https://")
        return _FakeUrlResp(payload)

    monkeypatch.setattr("inferencebench.commands.fetch.urllib.request.urlopen", fake_urlopen)

    uri = "https://example.com/results/envelope.json"
    result = runner.invoke(app, ["fetch", uri])
    assert result.exit_code == 0, result.output
    cached = list(isolated_cache.glob("*.json"))
    assert len(cached) == 1
    assert cached[0].read_bytes() == payload


def test_fetch_cache_hit_skips_redownload(
    signed_envelope_on_disk: Path,
    isolated_cache: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_count = {"n": 0}
    payload = signed_envelope_on_disk.read_bytes()

    def fake_urlopen(url: str) -> _FakeUrlResp:
        call_count["n"] += 1
        assert url.startswith("https://")
        return _FakeUrlResp(payload)

    monkeypatch.setattr("inferencebench.commands.fetch.urllib.request.urlopen", fake_urlopen)

    uri = "https://example.com/cached.json"
    first = runner.invoke(app, ["fetch", uri])
    assert first.exit_code == 0, first.output
    assert call_count["n"] == 1
    assert isolated_cache.exists()

    second = runner.invoke(app, ["fetch", uri])
    assert second.exit_code == 0, second.output
    # No new download happened.
    assert call_count["n"] == 1
    assert "cache hit" in second.output


def test_fetch_force_redownloads(
    signed_envelope_on_disk: Path,
    isolated_cache: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_count = {"n": 0}
    payload = signed_envelope_on_disk.read_bytes()

    def fake_urlopen(url: str) -> _FakeUrlResp:
        call_count["n"] += 1
        assert url.startswith("https://")
        return _FakeUrlResp(payload)

    monkeypatch.setattr("inferencebench.commands.fetch.urllib.request.urlopen", fake_urlopen)

    uri = "https://example.com/force.json"
    first = runner.invoke(app, ["fetch", uri])
    assert first.exit_code == 0, first.output
    assert isolated_cache.exists()

    second = runner.invoke(app, ["fetch", uri, "--force"])
    assert second.exit_code == 0, second.output
    assert call_count["n"] == 2


def test_fetch_invalid_json_exits_non_zero(
    tmp_path: Path,
    isolated_cache: Path,
) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{ not really json", encoding="utf-8")
    result = runner.invoke(app, ["fetch", str(bad)])
    assert result.exit_code != 0
    # File is left behind for debug.
    cached = list(isolated_cache.glob("*.json"))
    assert len(cached) == 1
    assert cached[0].read_text() == "{ not really json"


def test_fetch_invalid_envelope_schema_exits_non_zero(
    tmp_path: Path,
    isolated_cache: Path,
) -> None:
    bad = tmp_path / "not-an-envelope.json"
    bad.write_text(json.dumps({"some": "object"}), encoding="utf-8")
    result = runner.invoke(app, ["fetch", str(bad)])
    assert result.exit_code != 0
    assert isolated_cache.exists()


def test_fetch_hf_scheme_monkeypatched(
    signed_envelope_on_disk: Path,
    isolated_cache: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``hf://datasets/<owner>/<repo>`` defaults to ``envelope.json``."""
    seen: dict[str, Any] = {}

    def fake_hf_hub_download(
        *,
        repo_id: str,
        filename: str,
        repo_type: str,
    ) -> str:
        seen["repo_id"] = repo_id
        seen["filename"] = filename
        seen["repo_type"] = repo_type
        return str(signed_envelope_on_disk)

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_hf_hub_download)

    uri = "hf://datasets/Yobitel/llama-4-test"
    result = runner.invoke(app, ["fetch", uri])
    assert result.exit_code == 0, result.output

    assert seen["repo_id"] == "Yobitel/llama-4-test"
    assert seen["filename"] == "envelope.json"
    assert seen["repo_type"] == "dataset"

    cached = list(isolated_cache.glob("*.json"))
    assert len(cached) == 1


def test_fetch_hf_scheme_with_explicit_filename(
    signed_envelope_on_disk: Path,
    isolated_cache: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the file segment is present we forward it through."""
    seen: dict[str, Any] = {}

    def fake_hf_hub_download(
        *,
        repo_id: str,
        filename: str,
        repo_type: str,
    ) -> str:
        seen["repo_id"] = repo_id
        seen["filename"] = filename
        seen["repo_type"] = repo_type
        return str(signed_envelope_on_disk)

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_hf_hub_download)

    uri = "hf://datasets/Yobitel/llama-4-test/results/run-42.json"
    result = runner.invoke(app, ["fetch", uri])
    assert result.exit_code == 0, result.output
    assert seen["repo_id"] == "Yobitel/llama-4-test"
    assert seen["filename"] == "results/run-42.json"
    assert isolated_cache.exists()


def test_fetch_explicit_out_path_writes_there(
    tmp_path: Path,
    signed_envelope_on_disk: Path,
    isolated_cache: Path,
) -> None:
    out = tmp_path / "explicit" / "env.json"
    result = runner.invoke(app, ["fetch", str(signed_envelope_on_disk), "--out", str(out)])
    assert result.exit_code == 0, result.output
    assert out.exists()
    # Default cache should not be used when --out is provided.
    assert not isolated_cache.exists() or not list(isolated_cache.glob("*.json"))


def test_parse_hf_uri_rejects_missing_owner_repo() -> None:
    with pytest.raises(fetch_mod.FetchError):
        fetch_mod._parse_hf_uri("hf://datasets/only-one-part")


def test_parse_hf_uri_rejects_non_datasets() -> None:
    with pytest.raises(fetch_mod.FetchError):
        fetch_mod._parse_hf_uri("hf://models/owner/repo")
