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
    envelope_files = list((mirror / "llm-inference").glob("*.json"))
    assert len(envelope_files) == 1
    payload: dict[str, Any] = json.loads(envelope_files[0].read_text("utf-8"))
    assert payload["model"]["id"] == "meta-llama/Llama-4-Maverick"

    index = json.loads((mirror / "index.json").read_text("utf-8"))
    assert index["schema"] == "inferencebench.mirror.v1"
    assert index["n_entries"] == 1
    entry = index["entries"][0]
    assert entry["suite_id"] == "llm.inference"
    assert entry["model_id"] == "meta-llama/Llama-4-Maverick"
    assert entry["path"].startswith("llm-inference/")


def test_publish_local_index_appends_on_second_publish(tmp_path: Path) -> None:
    """Two publishes update the same index.json without overwriting prior entry."""
    env_path = _envelope_on_disk(tmp_path)
    mirror = tmp_path / "mirror"
    runner.invoke(app, ["publish", str(env_path), "--to", "local", "--workspace", str(mirror)])
    # publish a different envelope (different model id → different content_hash)
    other_env = make_envelope(
        model_id="mistralai/Mistral-Large",
        metrics={"throughput_tok_per_s": 900.0},
    )
    other_path = write_envelope_json(tmp_path / "other.json", other_env)
    runner.invoke(app, ["publish", str(other_path), "--to", "local", "--workspace", str(mirror)])

    index = json.loads((mirror / "index.json").read_text("utf-8"))
    assert index["n_entries"] == 2
    model_ids = {e["model_id"] for e in index["entries"]}
    assert model_ids == {"meta-llama/Llama-4-Maverick", "mistralai/Mistral-Large"}


def test_publish_unknown_target_errors(tmp_path: Path) -> None:
    env_path = _envelope_on_disk(tmp_path)
    result = runner.invoke(app, ["publish", str(env_path), "--to", "mars"])
    assert result.exit_code != 0


def test_publish_studio_target_is_phase2(tmp_path: Path) -> None:
    env_path = _envelope_on_disk(tmp_path)
    result = runner.invoke(app, ["publish", str(env_path), "--to", "studio"])
    assert result.exit_code != 0
    assert "Phase 2" in result.output or "Phase 2" in (result.stderr or "")
