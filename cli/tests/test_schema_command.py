"""Tests for ``bench schema``."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from inferencebench.cli import app

# Wide console so any rendered output doesn't wrap mid-token.
runner = CliRunner(env={"COLUMNS": "240"})


def test_default_envelope_schema_is_parseable_json() -> None:
    """``bench schema`` (default) emits pydantic-style JSON Schema."""
    result = runner.invoke(app, ["schema"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert isinstance(payload, dict)
    # Pydantic's model_json_schema always emits a top-level dict with at
    # least one of these structural keys.
    assert "$defs" in payload or "properties" in payload


def test_target_benchmark_spec() -> None:
    """``--target benchmark-spec`` works (plugin is installed in the workspace)."""
    result = runner.invoke(app, ["schema", "--target", "benchmark-spec"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert isinstance(payload, dict)
    assert payload.get("title") == "BenchmarkSpec" or "$defs" in payload or "properties" in payload


def test_target_mirror_index_is_valid_json_schema_dict() -> None:
    """``--target mirror-index`` emits a hand-written JSON Schema dict."""
    result = runner.invoke(app, ["schema", "--target", "mirror-index"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["$schema"].startswith("https://json-schema.org/")
    assert payload["type"] == "object"
    assert "entries" in payload["properties"]
    # The const matches the schema identifier the publisher writes.
    assert payload["properties"]["schema"]["const"] == "inferencebench.mirror.v1"
    # MirrorEntry definition is reachable.
    assert "MirrorEntry" in payload["$defs"]
    entry_props = payload["$defs"]["MirrorEntry"]["properties"]
    for required_field in (
        "suite_id",
        "suite_slug",
        "model_id",
        "engine",
        "content_hash",
        "path",
        "signed",
        "tag",
        "timestamp",
    ):
        assert required_field in entry_props, required_field


def test_version_flag_prints_v1() -> None:
    """``--version`` prints just the schema version string."""
    result = runner.invoke(app, ["schema", "--version"])
    assert result.exit_code == 0, result.output
    assert result.output.strip() == "v1"


def test_out_flag_writes_file_and_keeps_stdout_quiet(tmp_path: Path) -> None:
    """``--out`` writes the schema to a file and leaves stdout empty."""
    out_path = tmp_path / "out" / "envelope.schema.json"
    result = runner.invoke(app, ["schema", "--out", str(out_path)])
    assert result.exit_code == 0, result.output
    assert result.output.strip() == ""
    assert out_path.exists()
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    assert "$defs" in payload or "properties" in payload


def test_unknown_target_errors() -> None:
    """Unknown ``--target`` exits 2 with a red message."""
    result = runner.invoke(app, ["schema", "--target", "not-a-thing"])
    assert result.exit_code == 2
