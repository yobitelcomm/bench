"""Tests for ``bench profile`` — high-frequency telemetry re-run.

``profile`` shares the envelope-loading + plugin-discovery + signing plumbing
with ``replay``, so these tests focus on the profile-specific bits: the
mandatory ``--base-url`` error, the propagation of the 10 ms / 25 ms
telemetry intervals into ``RunContext.extra``, and the rendered profiling
breakdown table. The ``--no-verify`` bypass is exercised separately to keep
unsigned local fixtures useable.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from _helpers import (  # type: ignore[import-not-found]
    make_envelope,
    write_envelope_json,
)
from typer.testing import CliRunner

from inferencebench.cli import app

if TYPE_CHECKING:
    import pytest

runner = CliRunner(env={"COLUMNS": "240"})


# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #
def _sharegpt_envelope() -> Any:
    """An llm.inference.sharegpt-v3 envelope with a couple of headline metrics."""
    return make_envelope(
        model_id="meta-llama/Llama-3.1-8B-Instruct",
        run_id="01934567-89ab-7000-8000-0000000abcde",
        suite_id="llm.inference.sharegpt-v3",
        metrics={
            "throughput_tok_per_s": 1800.0,
            "ttft_p99_ms": 320.0,
            "ok_rate": 0.999,
        },
    )


# --------------------------------------------------------------------------- #
# Argument validation                                                         #
# --------------------------------------------------------------------------- #
def test_profile_missing_base_url_exits_two(tmp_path: Path) -> None:
    """Without --base-url we exit 2 with the same 'host-agnostic' rationale as replay."""
    env_path = write_envelope_json(tmp_path / "src.json", _sharegpt_envelope())
    result = runner.invoke(app, ["profile", str(env_path), "--no-verify"])
    assert result.exit_code == 2
    combined = result.stdout + (result.stderr or "")
    assert "--base-url" in combined
    assert "host-agnostic" in combined or "live engine" in combined


# --------------------------------------------------------------------------- #
# Happy path: telemetry overrides propagate                                   #
# --------------------------------------------------------------------------- #
def test_profile_happy_path_propagates_telemetry_intervals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dev_keypair: tuple[Path, Path]
) -> None:
    """Verify the profile-specific NVML=10 / RAPL=25 / duration=30 overrides reach
    ``RunContext.extra``, the new envelope is written, and the profiling
    breakdown table renders.
    """
    priv, _ = dev_keypair
    source_env = _sharegpt_envelope()
    source_path = write_envelope_json(tmp_path / "src.json", source_env)

    captured: dict[str, Any] = {}

    def _fake_run(_self: Any, _spec: Any, ctx: Any) -> Any:
        captured["model_id"] = ctx.model_id
        captured["engine_kind"] = ctx.engine_kind
        captured["base_url"] = ctx.base_url
        captured["extra"] = dict(ctx.extra)
        # Return a deterministic signed envelope — piggy-back on the dev key
        # so writer + downstream parsing round-trip cleanly.
        from inferencebench.envelope import SigningMode, sign_envelope

        new_env = make_envelope(
            model_id=ctx.model_id,
            run_id="01934567-89ab-7000-8000-0000000bbbbb",
            suite_id="llm.inference.sharegpt-v3",
            metrics={
                "throughput_tok_per_s": 1750.0,
                "ttft_p99_ms": 325.0,
                "ok_rate": 0.997,
                # Profiling-table fields — exercised by the breakdown renderer.
                "gpu_util_avg_pct": 78.5,
                "energy_joules_gpu": 1200.0,
                "energy_joules_cpu_dram": 300.0,
                "power_avg_w_under_load": 450.5,
                "nvml_sample_count": 3000,
                "rapl_sample_count": 1200,
            },
        )
        return sign_envelope(new_env, mode=SigningMode.DEV, dev_key_path=priv)

    from inferencebench_llm.plugin import LLMInferencePlugin

    monkeypatch.setattr(LLMInferencePlugin, "run", _fake_run)

    out_dir = tmp_path / "profile-out"
    result = runner.invoke(
        app,
        [
            "profile",
            str(source_path),
            "--base-url",
            "http://localhost:8000/v1",
            "--output",
            str(out_dir),
            "--dev-key",
            str(priv),
            "--no-verify",
        ],
    )
    assert result.exit_code == 0, result.stdout + (result.stderr or "")

    # Telemetry override knobs must reach RunContext.extra.
    assert captured["extra"]["nvml_interval_ms"] == 10
    assert captured["extra"]["rapl_interval_ms"] == 25
    # Default duration is 30 s for profile (not 300 s like run/replay).
    assert captured["extra"]["duration_s"] == 30
    # Source-envelope identity propagated.
    assert captured["model_id"] == "meta-llama/Llama-3.1-8B-Instruct"
    assert captured["engine_kind"].value == "vllm"
    assert captured["base_url"] == "http://localhost:8000/v1"

    # A new envelope file was written under --output.
    envelopes = list(out_dir.glob("*.json"))
    assert len(envelopes) == 1
    assert envelopes[0].name.startswith("profile-")

    # The profiling breakdown table rendered with the diagnostic columns.
    combined = result.stdout + (result.stderr or "")
    assert "Profiling breakdown" in combined
    assert "% time on host" in combined
    assert "Energy GPU vs CPU+DRAM" in combined
    assert "Avg power under load" in combined
    assert "NVML sample count" in combined
    assert "RAPL sample count" in combined


def test_profile_no_verify_bypasses_signature_checks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--no-verify`` lets unsigned local fixtures seed a profile run."""
    source_env = _sharegpt_envelope()  # unsigned
    source_path = write_envelope_json(tmp_path / "src.json", source_env)

    from inferencebench.envelope import (
        SigningMode,
        generate_dev_keypair,
        sign_envelope,
    )

    priv, _pub = generate_dev_keypair(tmp_path / "cosign.key")

    def _fake_run(_self: Any, _spec: Any, ctx: Any) -> Any:
        new_env = make_envelope(
            model_id=ctx.model_id,
            run_id="01934567-89ab-7000-8000-0000000ccccc",
            suite_id="llm.inference.sharegpt-v3",
            metrics={"throughput_tok_per_s": 1700.0, "ok_rate": 0.99},
        )
        return sign_envelope(new_env, mode=SigningMode.DEV, dev_key_path=priv)

    from inferencebench_llm.plugin import LLMInferencePlugin

    monkeypatch.setattr(LLMInferencePlugin, "run", _fake_run)

    out_dir = tmp_path / "profile-out"
    result = runner.invoke(
        app,
        [
            "profile",
            str(source_path),
            "--base-url",
            "http://localhost:8000/v1",
            "--output",
            str(out_dir),
            "--dev-key",
            str(priv),
            "--no-verify",
        ],
    )
    assert result.exit_code == 0, result.stdout + (result.stderr or "")
    assert len(list(out_dir.glob("*.json"))) == 1
