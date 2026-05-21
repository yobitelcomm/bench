"""Tests for ``bench plugin discover``.

The bundled registry is the source of truth for what's shipped in the
``inferencebench`` wheel; all tests load it via ``importlib.resources``
rather than reaching into the repo tree, so they exercise the same path
end-users hit after ``pip install inferencebench``.
"""

from __future__ import annotations

import json
from importlib import resources
from typing import TYPE_CHECKING

from typer.testing import CliRunner

from inferencebench.cli import app

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

runner = CliRunner()


def _load_bundled_registry() -> dict[str, object]:
    raw = (
        resources.files("inferencebench")
        .joinpath("data/plugin-registry.json")
        .read_text(encoding="utf-8")
    )
    return json.loads(raw)  # type: ignore[no-any-return]


def test_bundled_registry_parses_with_seven_entries() -> None:
    """The bundled registry is valid JSON and ships all 7 core plugins."""
    doc = _load_bundled_registry()
    assert doc["schema"] == "inferencebench.plugin-registry.v1"
    plugins = doc["plugins"]
    assert isinstance(plugins, list)
    assert len(plugins) == 7
    names = {p["name"] for p in plugins}
    assert names == {
        "llm.inference",
        "llm.quality",
        "llm.mt",
        "code.generation",
        "voice.transcription",
        "embeddings.retrieval",
        "vision.understanding",
    }
    # Every entry has the required fields.
    required = {
        "name",
        "package",
        "version",
        "install",
        "modality",
        "kind",
        "repo",
        "license",
        "status",
        "description",
        "engines_supported",
        "maintainer",
    }
    for entry in plugins:
        assert required <= set(entry), f"missing fields in {entry.get('name')}"


def test_discover_default_shows_all_seven_plugins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``bench plugin discover`` with no filters lists every plugin."""
    monkeypatch.setenv("BENCH_CACHE_ROOT", str(tmp_path / "cache"))
    result = runner.invoke(app, ["plugin", "discover"])
    assert result.exit_code == 0, result.stdout + (result.stderr or "")
    out = result.stdout
    for name in (
        "llm.inference",
        "llm.quality",
        "llm.mt",
        "code.generation",
        "voice.transcription",
        "embeddings.retrieval",
        "vision.understanding",
    ):
        assert name in out, f"missing plugin {name} in output:\n{out}"


def test_discover_modality_llm_filter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``--modality llm`` keeps only the llm.* plugins."""
    monkeypatch.setenv("BENCH_CACHE_ROOT", str(tmp_path / "cache"))
    result = runner.invoke(app, ["plugin", "discover", "--modality", "llm", "--json"])
    assert result.exit_code == 0, result.stdout + (result.stderr or "")
    doc = json.loads(result.stdout)
    names = {p["name"] for p in doc["plugins"]}
    assert names == {"llm.inference", "llm.quality"}


def test_discover_status_core_returns_all_seven(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("BENCH_CACHE_ROOT", str(tmp_path / "cache"))
    result = runner.invoke(app, ["plugin", "discover", "--status", "core", "--json"])
    assert result.exit_code == 0, result.stdout + (result.stderr or "")
    doc = json.loads(result.stdout)
    assert len(doc["plugins"]) == 7
    assert all(p["status"] == "core" for p in doc["plugins"])


def test_discover_installed_filter_lists_installed_plugins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """In the workspace dev env all 7 plugins are installed editable."""
    monkeypatch.setenv("BENCH_CACHE_ROOT", str(tmp_path / "cache"))
    result = runner.invoke(app, ["plugin", "discover", "--installed", "--json"])
    assert result.exit_code == 0, result.stdout + (result.stderr or "")
    doc = json.loads(result.stdout)
    names = {p["name"] for p in doc["plugins"]}
    assert names == {
        "llm.inference",
        "llm.quality",
        "llm.mt",
        "code.generation",
        "voice.transcription",
        "embeddings.retrieval",
        "vision.understanding",
    }


def test_discover_available_filter_is_empty_when_all_installed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With every plugin installed, ``--available`` lists nothing."""
    monkeypatch.setenv("BENCH_CACHE_ROOT", str(tmp_path / "cache"))
    result = runner.invoke(app, ["plugin", "discover", "--available", "--json"])
    assert result.exit_code == 0, result.stdout + (result.stderr or "")
    doc = json.loads(result.stdout)
    assert doc["plugins"] == []


def test_discover_json_has_expected_top_level_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("BENCH_CACHE_ROOT", str(tmp_path / "cache"))
    result = runner.invoke(app, ["plugin", "discover", "--json"])
    assert result.exit_code == 0, result.stdout + (result.stderr or "")
    doc = json.loads(result.stdout)
    assert {"schema", "updated_iso", "source", "plugins"} <= set(doc)
    assert doc["schema"] == "inferencebench.plugin-registry.v1"
    assert isinstance(doc["plugins"], list)


def test_discover_registry_loads_custom_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--registry <file>`` loads from the given file instead of the bundle."""
    monkeypatch.setenv("BENCH_CACHE_ROOT", str(tmp_path / "cache"))
    custom = tmp_path / "custom-registry.json"
    custom.write_text(
        json.dumps(
            {
                "schema": "inferencebench.plugin-registry.v1",
                "updated_iso": "2026-05-18",
                "plugins": [
                    {
                        "name": "custom.test",
                        "package": "custom-pkg",
                        "version": "9.9.9",
                        "install": "pip install custom-pkg",
                        "modality": "other",
                        "kind": "perf",
                        "repo": "https://example.invalid/custom",
                        "license": "MIT",
                        "status": "community",
                        "description": "Custom registry test entry.",
                        "engines_supported": [],
                        "maintainer": "tester",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    result = runner.invoke(app, ["plugin", "discover", "--registry", str(custom), "--json"])
    assert result.exit_code == 0, result.stdout + (result.stderr or "")
    doc = json.loads(result.stdout)
    assert [p["name"] for p in doc["plugins"]] == ["custom.test"]
    assert doc["source"].startswith("path:")


def test_discover_registry_bad_path_exits_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A nonexistent ``--registry`` path produces a clear error and exit 2."""
    monkeypatch.setenv("BENCH_CACHE_ROOT", str(tmp_path / "cache"))
    missing = tmp_path / "does-not-exist.json"
    result = runner.invoke(app, ["plugin", "discover", "--registry", str(missing)])
    assert result.exit_code == 2
    combined = result.stdout + (result.stderr or "")
    assert "failed to load registry" in combined
    assert "not found" in combined


def test_discover_rejects_invalid_modality(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BENCH_CACHE_ROOT", str(tmp_path / "cache"))
    result = runner.invoke(app, ["plugin", "discover", "--modality", "video"])
    assert result.exit_code == 2
    assert "invalid --modality" in result.stdout + (result.stderr or "")


def test_discover_installed_and_available_mutually_exclusive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("BENCH_CACHE_ROOT", str(tmp_path / "cache"))
    result = runner.invoke(app, ["plugin", "discover", "--installed", "--available"])
    assert result.exit_code == 2
    combined = result.stdout + (result.stderr or "")
    assert "mutually exclusive" in combined
