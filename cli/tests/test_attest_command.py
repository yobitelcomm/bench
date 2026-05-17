"""Tests for ``bench attest``."""

from __future__ import annotations

import json
from pathlib import Path

from _helpers import (  # type: ignore[import-not-found]
    make_envelope,
    write_envelope_json,
    write_signed_envelope_json,
)
from typer.testing import CliRunner

from inferencebench.cli import app

runner = CliRunner(env={"COLUMNS": "240"})


def _read_markdown(out_path: Path) -> str:
    assert out_path.exists(), f"attestation not written to {out_path}"
    return out_path.read_text(encoding="utf-8")


def test_attest_markdown_has_all_sections(
    tmp_path: Path, dev_keypair: tuple[Path, Path]
) -> None:
    priv, _pub = dev_keypair
    envelope = make_envelope(
        model_id="meta-llama/Llama-3.1-8B-Instruct",
        metrics={
            "throughput_tok_per_s": 1500.0,
            "ttft_p50_ms": 42.0,
            "ok_rate": 0.99,
        },
    )
    env_path = write_signed_envelope_json(tmp_path / "env.json", envelope, dev_key=priv)
    out_path = tmp_path / "attest.md"

    result = runner.invoke(
        app, ["attest", str(env_path), "--out", str(out_path), "--format", "markdown"]
    )
    assert result.exit_code == 0, result.output

    text = _read_markdown(out_path)
    # Seven sections from the spec.
    assert "# InferenceBench attestation" in text  # 1. Header
    assert "## What this is" in text  # 2.
    assert "## Subject" in text  # 3.
    assert "## Metrics" in text  # 4.
    assert "## Signature" in text  # 5.
    assert "## Verification" in text  # 6.
    assert "## About" in text  # 7. Footer
    # Verification block must include the literal verify command.
    assert "bench verify" in text


def test_attest_organization_in_header(
    tmp_path: Path, dev_keypair: tuple[Path, Path]
) -> None:
    priv, _pub = dev_keypair
    envelope = make_envelope(
        model_id="m",
        metrics={"throughput_tok_per_s": 1.0},
    )
    env_path = write_signed_envelope_json(tmp_path / "env.json", envelope, dev_key=priv)
    out_path = tmp_path / "attest.md"

    result = runner.invoke(
        app,
        [
            "attest",
            str(env_path),
            "--out",
            str(out_path),
            "--organization",
            "Acme",
        ],
    )
    assert result.exit_code == 0, result.output
    text = _read_markdown(out_path)
    assert "Issued for:" in text
    assert "Acme" in text


def test_attest_json_format_is_parseable(
    tmp_path: Path, dev_keypair: tuple[Path, Path]
) -> None:
    priv, _pub = dev_keypair
    envelope = make_envelope(
        model_id="m",
        metrics={"throughput_tok_per_s": 100.0, "ok_rate": 0.95},
    )
    env_path = write_signed_envelope_json(tmp_path / "env.json", envelope, dev_key=priv)
    out_path = tmp_path / "attest.json"

    result = runner.invoke(
        app,
        [
            "attest",
            str(env_path),
            "--out",
            str(out_path),
            "--format",
            "json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["schema"] == "inferencebench.attestation.v1"
    for key in ("header", "subject", "metrics", "signature", "verification"):
        assert key in payload, f"missing {key} in JSON attestation"


def test_attest_signed_envelope_renders_signature_method(
    tmp_path: Path, dev_keypair: tuple[Path, Path]
) -> None:
    priv, _pub = dev_keypair
    envelope = make_envelope(
        model_id="m", metrics={"throughput_tok_per_s": 1.0}
    )
    env_path = write_signed_envelope_json(tmp_path / "env.json", envelope, dev_key=priv)
    out_path = tmp_path / "attest.md"

    result = runner.invoke(
        app, ["attest", str(env_path), "--out", str(out_path)]
    )
    assert result.exit_code == 0, result.output
    text = _read_markdown(out_path)
    assert "dev-key" in text
    assert "Key fingerprint" in text


def test_attest_unsigned_envelope_renders_warning(tmp_path: Path) -> None:
    envelope = make_envelope(
        model_id="m", metrics={"throughput_tok_per_s": 1.0}
    )
    env_path = write_envelope_json(tmp_path / "env.json", envelope)
    out_path = tmp_path / "attest.md"

    result = runner.invoke(
        app, ["attest", str(env_path), "--out", str(out_path)]
    )
    assert result.exit_code == 0, result.output
    text = _read_markdown(out_path)
    assert "unsigned" in text.lower()
    # The CLI itself should print a yellow warning to the user.
    assert "Warning" in result.output or "warning" in result.output


def test_attest_json_content_hash_matches_envelope(
    tmp_path: Path, dev_keypair: tuple[Path, Path]
) -> None:
    priv, _pub = dev_keypair
    envelope = make_envelope(
        model_id="m", metrics={"throughput_tok_per_s": 1.0}
    )
    env_path = write_signed_envelope_json(tmp_path / "env.json", envelope, dev_key=priv)
    out_path = tmp_path / "attest.json"

    result = runner.invoke(
        app,
        [
            "attest",
            str(env_path),
            "--out",
            str(out_path),
            "--format",
            "json",
        ],
    )
    assert result.exit_code == 0, result.output

    source_hash = json.loads(env_path.read_text(encoding="utf-8"))
    # Recompute via the public model API so the test exercises canonical hashing.
    from inferencebench.envelope import Envelope

    source_envelope = Envelope.model_validate(source_hash)
    expected = source_envelope.content_hash()

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["header"]["content_hash"] == expected
