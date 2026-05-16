"""Tests for ``bench publish`` (ticket 0030)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from _helpers import make_envelope, write_envelope_json  # type: ignore[import-not-found]
from typer.testing import CliRunner

from inferencebench.cli import app

runner = CliRunner(env={"COLUMNS": "240"})


def _envelope_on_disk(tmp_path: Path) -> Path:
    env = make_envelope(
        model_id="meta-llama/Llama-4-Maverick",
        metrics={"throughput_tok_per_s": 1500.0, "ttft_p99_ms": 400.0},
    )
    return write_envelope_json(tmp_path / "envelope.json", env)


def test_publish_missing_envelope_errors(tmp_path: Path) -> None:
    result = runner.invoke(app, ["publish", str(tmp_path / "nope.json")])
    assert result.exit_code != 0


def test_publish_hf_dry_run_does_not_touch_network(tmp_path: Path) -> None:
    env_path = _envelope_on_disk(tmp_path)
    result = runner.invoke(
        app,
        ["publish", str(env_path), "--to", "hf", "--dry-run", "--org", "test-org"],
    )
    assert result.exit_code == 0, result.output
    assert "DRY-RUN" in result.output
    assert "test-org" in result.output


def test_publish_local_copies_to_mirror_dir(tmp_path: Path) -> None:
    env_path = _envelope_on_disk(tmp_path)
    mirror = tmp_path / "mirror"
    result = runner.invoke(
        app,
        ["publish", str(env_path), "--to", "local", "--workspace", str(mirror)],
    )
    assert result.exit_code == 0, result.output
    written = list(mirror.rglob("*.json"))
    assert len(written) == 1
    payload: dict[str, Any] = json.loads(written[0].read_text("utf-8"))
    assert payload["model"]["id"] == "meta-llama/Llama-4-Maverick"


def test_publish_unknown_target_errors(tmp_path: Path) -> None:
    env_path = _envelope_on_disk(tmp_path)
    result = runner.invoke(app, ["publish", str(env_path), "--to", "mars"])
    assert result.exit_code != 0


def test_publish_studio_target_is_phase2(tmp_path: Path) -> None:
    env_path = _envelope_on_disk(tmp_path)
    result = runner.invoke(app, ["publish", str(env_path), "--to", "studio"])
    assert result.exit_code != 0
    assert "Phase 2" in result.output or "Phase 2" in (result.stderr or "")
