"""Tests for ``bench replay`` — envelope-driven re-run command.

The replay command consumes an existing envelope and re-runs the same
suite/model/engine/dataset/seed against a fresh ``--base-url``. Most of the
machinery (plugin discovery, RunContext construction, signing-extras) is
shared with ``bench run`` and is exercised there; these tests focus on the
replay-specific wiring: envelope loading, source-envelope verification,
suite_id resolution, and propagation of source fields into the RunContext.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from _helpers import (  # type: ignore[import-not-found]
    make_envelope,
    write_envelope_json,
    write_signed_envelope_json,
)
from typer.testing import CliRunner

from inferencebench.cli import app
from inferencebench.commands import replay as replay_module

if TYPE_CHECKING:
    import pytest

runner = CliRunner(env={"COLUMNS": "240"})


# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #
def _sharegpt_envelope() -> Any:
    """An llm.inference.sharegpt-v3 envelope with the canonical metrics."""
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
def test_replay_missing_envelope_path() -> None:
    """Typer rejects the call when the positional envelope_path is missing."""
    result = runner.invoke(app, ["replay"])
    assert result.exit_code == 2
    combined = result.stdout + (result.stderr or "")
    # Typer's "missing argument" error.
    assert "ENVELOPE_PATH" in combined or "Missing argument" in combined


def test_replay_nonexistent_envelope_path(tmp_path: Path) -> None:
    """An envelope path that doesn't exist exits 2 with a clear error."""
    result = runner.invoke(app, ["replay", str(tmp_path / "does-not-exist.json"), "--no-verify"])
    assert result.exit_code == 2
    combined = result.stdout + (result.stderr or "")
    assert "not found" in combined.lower()


def test_replay_missing_base_url_errors(tmp_path: Path) -> None:
    """Without --base-url we exit 2 and explain why the envelope can't supply it."""
    env_path = write_envelope_json(tmp_path / "src.json", _sharegpt_envelope())
    result = runner.invoke(app, ["replay", str(env_path), "--no-verify"])
    assert result.exit_code == 2
    combined = result.stdout + (result.stderr or "")
    assert "--base-url" in combined
    # The message must explain WHY the envelope doesn't supply it.
    assert "host-agnostic" in combined or "live engine" in combined


# --------------------------------------------------------------------------- #
# Plugin / suite_id resolution                                                #
# --------------------------------------------------------------------------- #
def test_replay_unknown_suite_id_errors(tmp_path: Path) -> None:
    """An envelope whose suite_id isn't shipped by any plugin exits 1."""
    env = make_envelope(
        model_id="meta-llama/Llama-3.1-8B-Instruct",
        run_id="01934567-89ab-7000-8000-0000000fffff",
        suite_id="llm.inference.does-not-exist",
        metrics={"throughput_tok_per_s": 1000.0},
    )
    env_path = write_envelope_json(tmp_path / "src.json", env)
    result = runner.invoke(
        app,
        [
            "replay",
            str(env_path),
            "--base-url",
            "http://localhost:8000/v1",
            "--no-verify",
        ],
    )
    assert result.exit_code == 1
    combined = result.stdout + (result.stderr or "")
    assert "llm.inference.does-not-exist" in combined
    # The error should make the cause obvious.
    assert "no longer ships" in combined.lower() or "not" in combined.lower()


# --------------------------------------------------------------------------- #
# Happy path with monkeypatched plugin.run                                    #
# --------------------------------------------------------------------------- #
def test_replay_happy_path_propagates_source_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dev_keypair: tuple[Path, Path]
) -> None:
    """The replay propagates model.id, engine.name and quantization from the source
    envelope into the RunContext handed to the plugin, and writes the new envelope.
    """
    priv, _ = dev_keypair
    source_env = _sharegpt_envelope()
    source_path = write_signed_envelope_json(tmp_path / "src.json", source_env, dev_key=priv)

    captured: dict[str, Any] = {}

    def _fake_run(_self: Any, spec: Any, ctx: Any) -> Any:
        captured["spec_id"] = spec.benchmark_id
        captured["model_id"] = ctx.model_id
        captured["engine_kind"] = ctx.engine_kind
        captured["base_url"] = ctx.base_url
        captured["quantization_format"] = ctx.quantization_format
        captured["output_dir"] = ctx.output_dir
        captured["extra"] = dict(ctx.extra)
        # Return a deterministic "fake-signed" replay envelope. We piggy-back on
        # the existing dev-signing path so the envelope round-trips through the
        # standard writer + verifier.
        from inferencebench.envelope import SigningMode, sign_envelope

        replay_env = make_envelope(
            model_id=ctx.model_id,
            run_id="01934567-89ab-7000-8000-0000000bbbbb",
            suite_id="llm.inference.sharegpt-v3",
            metrics={
                "throughput_tok_per_s": 1820.5,
                "ttft_p99_ms": 318.0,
                "ok_rate": 0.998,
            },
        )
        return sign_envelope(replay_env, mode=SigningMode.DEV, dev_key_path=priv)

    from inferencebench_llm.plugin import LLMInferencePlugin

    monkeypatch.setattr(LLMInferencePlugin, "run", _fake_run)

    out_dir = tmp_path / "replay-out"
    result = runner.invoke(
        app,
        [
            "replay",
            str(source_path),
            "--base-url",
            "http://localhost:8000/v1",
            "--output",
            str(out_dir),
            "--dev-key",
            str(priv),
        ],
    )
    assert result.exit_code == 0, result.stdout + (result.stderr or "")

    # The fake plugin.run got the source envelope's identity.
    assert captured["spec_id"] == "llm.inference.sharegpt-v3"
    assert captured["model_id"] == "meta-llama/Llama-3.1-8B-Instruct"
    assert captured["engine_kind"].value == "vllm"
    assert captured["base_url"] == "http://localhost:8000/v1"
    # Source envelope was quant=fp8 (see _helpers.make_envelope).
    assert captured["quantization_format"] == "fp8"
    assert captured["output_dir"] == out_dir
    # Signing extras forwarded.
    assert captured["extra"]["signing_mode"] == "dev"
    assert captured["extra"]["dev_key_path"] == str(priv)

    # A new envelope file was written under --output.
    envelopes = list(out_dir.glob("*.json"))
    assert len(envelopes) == 1
    new_env = json.loads(envelopes[0].read_text())
    assert new_env["suite_id"] == "llm.inference.sharegpt-v3"
    assert new_env["model"]["id"] == "meta-llama/Llama-3.1-8B-Instruct"

    # The summary table must show both envelope paths.
    assert str(source_path) in result.stdout or str(source_path) in (result.stderr or "")


def test_replay_no_verify_bypasses_signature_checks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--no-verify lets unsigned local fixtures seed a replay."""
    source_env = _sharegpt_envelope()  # unsigned
    source_path = write_envelope_json(tmp_path / "src.json", source_env)

    # Generate a fresh dev key for signing the replay output.
    from inferencebench.envelope import (
        SigningMode,
        generate_dev_keypair,
        sign_envelope,
    )

    priv, _pub = generate_dev_keypair(tmp_path / "cosign.key")

    def _fake_run(_self: Any, _spec: Any, ctx: Any) -> Any:
        replay_env = make_envelope(
            model_id=ctx.model_id,
            run_id="01934567-89ab-7000-8000-0000000ccccc",
            suite_id="llm.inference.sharegpt-v3",
            metrics={"throughput_tok_per_s": 1700.0, "ok_rate": 0.99},
        )
        return sign_envelope(replay_env, mode=SigningMode.DEV, dev_key_path=priv)

    from inferencebench_llm.plugin import LLMInferencePlugin

    monkeypatch.setattr(LLMInferencePlugin, "run", _fake_run)

    out_dir = tmp_path / "replay-out"
    result = runner.invoke(
        app,
        [
            "replay",
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


def test_replay_verify_rejects_unsigned_envelope(tmp_path: Path) -> None:
    """Default --verify=True refuses unsigned envelopes (which can't be verified)."""
    env_path = write_envelope_json(tmp_path / "src.json", _sharegpt_envelope())
    result = runner.invoke(
        app,
        [
            "replay",
            str(env_path),
            "--base-url",
            "http://localhost:8000/v1",
        ],
    )
    assert result.exit_code == 1
    combined = result.stdout + (result.stderr or "")
    assert "verif" in combined.lower()


# --------------------------------------------------------------------------- #
# Module-level helper smoke test                                              #
# --------------------------------------------------------------------------- #
def test_replay_loader_rejects_remote_uris() -> None:
    """Phase 1 supports local paths only — remote URIs exit 2."""
    result = runner.invoke(
        app,
        [
            "replay",
            "hf://datasets/foo/bar.json",
            "--base-url",
            "http://localhost:8000/v1",
            "--no-verify",
        ],
    )
    assert result.exit_code == 2
    combined = result.stdout + (result.stderr or "")
    assert "hf://" in combined or "local file" in combined.lower()


def test_replay_module_has_helpers() -> None:
    """Sanity: the new module exposes the expected helpers for reuse."""
    assert hasattr(replay_module, "replay")
    assert hasattr(replay_module, "_load_envelope")
