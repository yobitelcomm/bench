"""Tests for ``bench list`` (top-level benchmark catalogue).

These verify the cross-plugin listing behaviour: full discovery via the
``inferencebench.plugins`` entry-point group, ``--plugin`` filtering,
graceful empty-plugins handling, and ``--json`` output.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from typer.testing import CliRunner

from inferencebench.cli import app
from inferencebench.commands import list_cmd

if TYPE_CHECKING:
    import pytest

runner = CliRunner()


_BUNDLED_BENCHMARKS = {
    "llm.inference.sharegpt-v3",
    "llm.inference.chatbot-short",
    "llm.inference.long-context",
}


def test_list_shows_bundled_llm_inference_benchmarks() -> None:
    """`bench list` lists every benchmark from every installed plugin."""
    result = runner.invoke(app, ["list"])
    assert result.exit_code == 0, result.stdout + (result.stderr or "")
    for bench_id in _BUNDLED_BENCHMARKS:
        assert bench_id in result.stdout, f"missing {bench_id}"
    # Plugin column shows up too.
    assert "llm.inference" in result.stdout


def test_list_plugin_filter_works() -> None:
    """`--plugin llm.inference` shows that plugin's benchmarks and only those."""
    result = runner.invoke(app, ["list", "--plugin", "llm.inference"])
    assert result.exit_code == 0, result.stdout + (result.stderr or "")
    for bench_id in _BUNDLED_BENCHMARKS:
        assert bench_id in result.stdout


def test_list_unknown_plugin_filter_errors() -> None:
    """`--plugin nonexistent` exits non-zero with a clear error."""
    result = runner.invoke(app, ["list", "--plugin", "nonexistent"])
    assert result.exit_code != 0
    combined = result.stdout + (result.stderr or "")
    assert "nonexistent" in combined
    assert "no plugin" in combined.lower()


def test_list_json_emits_documented_structure() -> None:
    """`--json` emits parseable JSON keyed by plugin → version + benchmarks list."""
    result = runner.invoke(app, ["list", "--json"])
    assert result.exit_code == 0, result.stdout + (result.stderr or "")
    payload = json.loads(result.stdout)
    assert "plugins" in payload
    assert "llm.inference" in payload["plugins"]
    plugin_entry = payload["plugins"]["llm.inference"]
    assert "version" in plugin_entry
    assert "benchmarks" in plugin_entry
    ids = {b["benchmark_id"] for b in plugin_entry["benchmarks"]}
    assert _BUNDLED_BENCHMARKS.issubset(ids)
    # Spec dicts should include the expected canonical fields.
    sample = next(
        b for b in plugin_entry["benchmarks"]
        if b["benchmark_id"] == "llm.inference.sharegpt-v3"
    )
    for key in ("modality", "kind", "dataset", "driver"):
        assert key in sample, f"missing key in spec dict: {key}"


def test_list_empty_plugins_message(monkeypatch: pytest.MonkeyPatch) -> None:
    """With 0 plugins discovered, exit 0 and print a yellow hint."""
    monkeypatch.setattr(list_cmd, "_entry_points", lambda: [])

    result = runner.invoke(app, ["list"])
    assert result.exit_code == 0, result.stdout + (result.stderr or "")
    assert "No plugins installed" in result.stdout


def test_list_empty_plugins_json(monkeypatch: pytest.MonkeyPatch) -> None:
    """`--json` on the empty-plugins case still emits the documented shape."""
    monkeypatch.setattr(list_cmd, "_entry_points", lambda: [])

    result = runner.invoke(app, ["list", "--json"])
    assert result.exit_code == 0, result.stdout + (result.stderr or "")
    payload = json.loads(result.stdout)
    assert payload == {"plugins": {}}
