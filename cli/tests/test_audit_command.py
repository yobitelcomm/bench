"""Tests for ``bench audit``."""

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


def test_audit_passes_signed_envelopes(
    tmp_path: Path, dev_keypair: tuple[Path, Path]
) -> None:
    priv, pub = dev_keypair
    e1 = make_envelope(
        model_id="meta-llama/Llama-3.1-8B-Instruct",
        run_id="01934567-89ab-7000-8000-000000000001",
        metrics={"throughput_tok_per_s": 1500.0},
    )
    e2 = make_envelope(
        model_id="Qwen/Qwen2.5-7B-Instruct",
        run_id="01934567-89ab-7000-8000-000000000002",
        metrics={"throughput_tok_per_s": 1300.0},
    )
    write_signed_envelope_json(tmp_path / "a.json", e1, dev_key=priv)
    write_signed_envelope_json(tmp_path / "b.json", e2, dev_key=priv)

    result = runner.invoke(
        app, ["audit", str(tmp_path), "--dev-public-key", str(pub)]
    )
    assert result.exit_code == 0, result.output
    assert "2 / 2 envelopes verified" in result.output


def test_audit_flags_unsigned_envelope(tmp_path: Path) -> None:
    e1 = make_envelope(model_id="m", metrics={"throughput_tok_per_s": 1.0})
    write_envelope_json(tmp_path / "a.json", e1)
    result = runner.invoke(app, ["audit", str(tmp_path)])
    assert result.exit_code == 1
    assert "no signature" in result.output


def test_audit_flags_tampered_envelope(
    tmp_path: Path, dev_keypair: tuple[Path, Path]
) -> None:
    priv, pub = dev_keypair
    e1 = make_envelope(model_id="m", metrics={"throughput_tok_per_s": 1.0})
    path = write_signed_envelope_json(tmp_path / "a.json", e1, dev_key=priv)
    raw = json.loads(path.read_text("utf-8"))
    raw["metrics"]["throughput_tok_per_s"] = 9999.0
    path.write_text(json.dumps(raw, sort_keys=True, indent=2), encoding="utf-8")

    result = runner.invoke(
        app, ["audit", str(tmp_path), "--dev-public-key", str(pub)]
    )
    assert result.exit_code == 1


def test_audit_json_report(
    tmp_path: Path, dev_keypair: tuple[Path, Path]
) -> None:
    priv, pub = dev_keypair
    e1 = make_envelope(model_id="m", metrics={"throughput_tok_per_s": 1.0})
    write_signed_envelope_json(tmp_path / "a.json", e1, dev_key=priv)

    result = runner.invoke(
        app,
        ["audit", str(tmp_path), "--dev-public-key", str(pub), "--report", "json"],
    )
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["schema"] == "inferencebench.audit.v1"
    assert payload["n_total"] == 1
    assert payload["n_ok"] == 1


def test_audit_no_strict_returns_zero_even_on_failure(tmp_path: Path) -> None:
    e1 = make_envelope(model_id="m", metrics={"throughput_tok_per_s": 1.0})
    write_envelope_json(tmp_path / "a.json", e1)
    result = runner.invoke(app, ["audit", str(tmp_path), "--no-strict"])
    assert result.exit_code == 0


def test_audit_skips_samples_files(
    tmp_path: Path, dev_keypair: tuple[Path, Path]
) -> None:
    priv, pub = dev_keypair
    e1 = make_envelope(model_id="m", metrics={"throughput_tok_per_s": 1.0})
    write_signed_envelope_json(tmp_path / "a.json", e1, dev_key=priv)
    (tmp_path / "samples-12345.jsonl").write_text(
        '{"request_idx": 0}\n', encoding="utf-8"
    )
    (tmp_path / "samples-67890.json").write_text(
        '{"request_idx": 0}\n', encoding="utf-8"
    )
    result = runner.invoke(
        app, ["audit", str(tmp_path), "--dev-public-key", str(pub)]
    )
    assert result.exit_code == 0
    assert "1 / 1 envelopes verified" in result.output


def test_audit_single_file(
    tmp_path: Path, dev_keypair: tuple[Path, Path]
) -> None:
    priv, pub = dev_keypair
    e1 = make_envelope(model_id="m", metrics={"throughput_tok_per_s": 1.0})
    path = write_signed_envelope_json(tmp_path / "single.json", e1, dev_key=priv)
    result = runner.invoke(
        app, ["audit", str(path), "--dev-public-key", str(pub)]
    )
    assert result.exit_code == 0
    assert "1 / 1" in result.output
