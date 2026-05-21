"""Tests for ``bench watch`` — continuous leaderboard rebuilds."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import pytest
from _helpers import make_envelope, write_envelope_json  # type: ignore[import-not-found]
from typer.testing import CliRunner

from inferencebench.cli import app
from inferencebench.commands import watch as watch_module

runner = CliRunner(env={"COLUMNS": "240"})


def _envelope(model_id: str, run_suffix: str) -> Any:
    return make_envelope(
        model_id=model_id,
        run_id=f"01934567-89ab-7000-8000-{run_suffix:>012}",
        metrics={
            "throughput_tok_per_s": 1500.0,
            "ttft_p99_ms": 400.0,
            "cost_usd_per_million_tokens": 0.5,
        },
    )


def test_watch_empty_dir_initial_build(tmp_path: Path) -> None:
    """Even when no envelopes exist yet, the first iteration renders a site."""
    envelopes_dir = tmp_path / "envelopes"
    envelopes_dir.mkdir()
    out_dir = tmp_path / "site"

    result = runner.invoke(
        app,
        [
            "watch",
            str(envelopes_dir),
            "--out",
            str(out_dir),
            "--max-iterations",
            "1",
            "--interval-s",
            "0.1",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (out_dir / "index.html").exists()
    assert "rebuilt site" in result.output


def test_watch_picks_up_new_envelope_between_polls(tmp_path: Path) -> None:
    """A second envelope written between polls is reflected in the rebuilt site."""
    envelopes_dir = tmp_path / "envelopes"
    envelopes_dir.mkdir()
    out_dir = tmp_path / "site"

    write_envelope_json(
        envelopes_dir / "a.json",
        _envelope("meta-llama/Llama-4-Maverick", "1"),
    )

    second_path = envelopes_dir / "b.json"
    second_envelope = _envelope("mistralai/Mistral-Large", "2")

    timer = threading.Timer(0.05, lambda: write_envelope_json(second_path, second_envelope))
    timer.start()

    try:
        result = runner.invoke(
            app,
            [
                "watch",
                str(envelopes_dir),
                "--out",
                str(out_dir),
                "--max-iterations",
                "2",
                "--interval-s",
                "0.2",
            ],
        )
    finally:
        timer.cancel()
        timer.join()

    assert result.exit_code == 0, result.output
    assert (out_dir / "index.html").exists()

    leaderboard_json = out_dir / "data" / "leaderboard.json"
    assert leaderboard_json.exists(), result.output
    payload = json.loads(leaderboard_json.read_text())

    total_entries = sum(len(c["entries"]) for c in payload["categories"])
    assert total_entries == 2, payload


def test_watch_max_iterations_runs_exactly_n(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--max-iterations 3`` produces exactly 3 poll cycles."""
    envelopes_dir = tmp_path / "envelopes"
    envelopes_dir.mkdir()
    out_dir = tmp_path / "site"

    sleeps: list[float] = []
    real_sleep = watch_module.time.sleep

    def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        real_sleep(0)

    monkeypatch.setattr(watch_module.time, "sleep", fake_sleep)

    result = runner.invoke(
        app,
        [
            "watch",
            str(envelopes_dir),
            "--out",
            str(out_dir),
            "--max-iterations",
            "3",
            "--interval-s",
            "0.01",
        ],
    )
    assert result.exit_code == 0, result.output
    # N-1 sleeps for N iterations (no sleep after the last iteration before exit).
    assert len(sleeps) == 2, sleeps


def test_watch_invalid_dir_exits_2(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist"
    out_dir = tmp_path / "site"
    result = runner.invoke(
        app,
        [
            "watch",
            str(missing),
            "--out",
            str(out_dir),
            "--max-iterations",
            "1",
        ],
    )
    assert result.exit_code == 2, result.output


def test_watch_render_failure_keeps_looping_and_exits_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If ``render_site`` raises, the loop continues and final exit code is 1."""
    envelopes_dir = tmp_path / "envelopes"
    envelopes_dir.mkdir()
    out_dir = tmp_path / "site"

    import inferencebench_leaderboard

    call_count = {"n": 0}

    def boom(*_args: Any, **_kwargs: Any) -> None:
        call_count["n"] += 1
        raise RuntimeError("synthetic render failure")

    monkeypatch.setattr(inferencebench_leaderboard, "render_site", boom)

    result = runner.invoke(
        app,
        [
            "watch",
            str(envelopes_dir),
            "--out",
            str(out_dir),
            "--max-iterations",
            "1",
            "--interval-s",
            "0.01",
        ],
    )
    assert result.exit_code == 1, result.output
    assert call_count["n"] >= 1
    assert "render_site failed" in result.output
