"""Tests for ``bench bundle create`` / ``bench bundle extract``."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import pytest
from _helpers import (  # type: ignore[import-not-found]
    make_envelope,
    write_envelope_json,
    write_signed_envelope_json,
)
from typer.testing import CliRunner

from inferencebench.cli import app
from inferencebench.envelope import Envelope
from inferencebench.envelope.models import Signature

runner = CliRunner()


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def _signed_envelope_on_disk(tmp_path: Path, dev_keypair: tuple[Path, Path]) -> tuple[Path, Path]:
    """Write a signed envelope.json to tmp; return (envelope_path, pubkey_path)."""
    priv, pub = dev_keypair
    env = make_envelope(model_id="meta-llama/Llama-4", metrics={"ttft_p50_ms": 12.3})
    env_path = tmp_path / "envelope.json"
    write_signed_envelope_json(env_path, env, dev_key=priv)
    return env_path, pub


# --------------------------------------------------------------------------- #
# bundle create                                                               #
# --------------------------------------------------------------------------- #
def test_bundle_create_contains_expected_files(
    tmp_path: Path, dev_keypair: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A vanilla `bundle create` produces a zip with the four required entries."""
    monkeypatch.chdir(tmp_path)
    env_path, _ = _signed_envelope_on_disk(tmp_path, dev_keypair)

    result = runner.invoke(app, ["bundle", "create", str(env_path)])
    assert result.exit_code == 0, result.stdout + (result.stderr or "")

    zips = list(tmp_path.glob("*.bundle.zip"))
    assert len(zips) == 1, f"expected exactly one bundle zip, found: {zips}"
    bundle_zip = zips[0]

    with zipfile.ZipFile(bundle_zip) as zf:
        names = set(zf.namelist())
    assert names == {"envelope.json", "signature_info.json", "verify.py", "README.txt"}


def test_bundle_create_signature_info_payload(
    tmp_path: Path, dev_keypair: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """signature_info.json mirrors the signature block of the envelope."""
    monkeypatch.chdir(tmp_path)
    env_path, _ = _signed_envelope_on_disk(tmp_path, dev_keypair)
    result = runner.invoke(app, ["bundle", "create", str(env_path)])
    assert result.exit_code == 0

    bundle_zip = next(tmp_path.glob("*.bundle.zip"))
    with zipfile.ZipFile(bundle_zip) as zf:
        info = json.loads(zf.read("signature_info.json").decode("utf-8"))
    assert info["method"] == "dev-key"
    assert info["bundle_present"] is True
    assert isinstance(info["content_hash"], str)
    assert len(info["content_hash"]) == 64
    assert isinstance(info["key_id"], str)


def test_bundle_create_with_public_key(
    tmp_path: Path, dev_keypair: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--include-public-key` adds cosign.pub to the zip."""
    monkeypatch.chdir(tmp_path)
    env_path, pub_path = _signed_envelope_on_disk(tmp_path, dev_keypair)

    result = runner.invoke(
        app,
        ["bundle", "create", str(env_path), "--include-public-key", str(pub_path)],
    )
    assert result.exit_code == 0, result.stdout + (result.stderr or "")

    bundle_zip = next(tmp_path.glob("*.bundle.zip"))
    with zipfile.ZipFile(bundle_zip) as zf:
        names = set(zf.namelist())
    assert "cosign.pub" in names


def test_bundle_create_picks_up_neighbouring_samples(
    tmp_path: Path, dev_keypair: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--include-samples` (default) bundles samples-*.jsonl files near the envelope."""
    monkeypatch.chdir(tmp_path)
    env_path, _ = _signed_envelope_on_disk(tmp_path, dev_keypair)

    samples_a = tmp_path / "samples-001.jsonl"
    samples_a.write_text('{"request_idx": 0}\n', encoding="utf-8")
    samples_b = tmp_path / "samples-002.jsonl"
    samples_b.write_text('{"request_idx": 1}\n', encoding="utf-8")
    # Make sure mtimes are within the 5-minute window of the envelope mtime.
    now = time.time()
    os.utime(env_path, (now, now))
    os.utime(samples_a, (now, now))
    os.utime(samples_b, (now, now))

    result = runner.invoke(app, ["bundle", "create", str(env_path)])
    assert result.exit_code == 0, result.stdout + (result.stderr or "")

    bundle_zip = next(tmp_path.glob("*.bundle.zip"))
    with zipfile.ZipFile(bundle_zip) as zf:
        names = set(zf.namelist())
        samples_text = zf.read("samples.jsonl").decode("utf-8")
    assert "samples.jsonl" in names
    assert '"request_idx": 0' in samples_text
    assert '"request_idx": 1' in samples_text


def test_bundle_create_no_samples_flag(
    tmp_path: Path, dev_keypair: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--no-include-samples` skips neighbouring samples files."""
    monkeypatch.chdir(tmp_path)
    env_path, _ = _signed_envelope_on_disk(tmp_path, dev_keypair)
    (tmp_path / "samples-001.jsonl").write_text('{"a": 1}\n', encoding="utf-8")

    result = runner.invoke(app, ["bundle", "create", str(env_path), "--no-include-samples"])
    assert result.exit_code == 0

    bundle_zip = next(tmp_path.glob("*.bundle.zip"))
    with zipfile.ZipFile(bundle_zip) as zf:
        assert "samples.jsonl" not in zf.namelist()


def test_bundle_create_rejects_invalid_envelope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Bundling an invalid envelope exits with code 2."""
    monkeypatch.chdir(tmp_path)
    bad = tmp_path / "bad.json"
    bad.write_text('{"not": "an envelope"}', encoding="utf-8")

    result = runner.invoke(app, ["bundle", "create", str(bad)])
    assert result.exit_code == 2


def test_bundle_create_rejects_missing_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Missing envelope path exits with code 2."""
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["bundle", "create", str(tmp_path / "nope.json")])
    assert result.exit_code == 2


# --------------------------------------------------------------------------- #
# bundle extract                                                              #
# --------------------------------------------------------------------------- #
def test_bundle_extract_roundtrips(
    tmp_path: Path, dev_keypair: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """pack → unpack → reload envelope works end-to-end."""
    monkeypatch.chdir(tmp_path)
    env_path, _ = _signed_envelope_on_disk(tmp_path, dev_keypair)

    create_result = runner.invoke(app, ["bundle", "create", str(env_path)])
    assert create_result.exit_code == 0
    bundle_zip = next(tmp_path.glob("*.bundle.zip"))

    extract_dir = tmp_path / "unpacked"
    extract_result = runner.invoke(
        app, ["bundle", "extract", str(bundle_zip), "--out", str(extract_dir)]
    )
    assert extract_result.exit_code == 0, extract_result.stdout + (extract_result.stderr or "")

    loaded = Envelope.model_validate(
        json.loads((extract_dir / "envelope.json").read_text(encoding="utf-8"))
    )
    original = Envelope.model_validate(json.loads(env_path.read_text(encoding="utf-8")))
    assert loaded.content_hash() == original.content_hash()


def test_bundle_extract_default_out_dir(
    tmp_path: Path, dev_keypair: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without --out, extract writes to ./<basename-without-.bundle.zip>/."""
    monkeypatch.chdir(tmp_path)
    env_path, _ = _signed_envelope_on_disk(tmp_path, dev_keypair)

    runner.invoke(app, ["bundle", "create", str(env_path), "--out", str(tmp_path / "x.bundle.zip")])
    result = runner.invoke(app, ["bundle", "extract", str(tmp_path / "x.bundle.zip")])
    assert result.exit_code == 0
    assert (tmp_path / "x" / "envelope.json").exists()


# --------------------------------------------------------------------------- #
# Standalone verify.py — the load-bearing piece                               #
# --------------------------------------------------------------------------- #
def test_standalone_verify_script_succeeds(
    tmp_path: Path, dev_keypair: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Run the bundled verify.py on a real signed bundle — expect exit 0 + OK."""
    monkeypatch.chdir(tmp_path)
    env_path, pub_path = _signed_envelope_on_disk(tmp_path, dev_keypair)

    bundle_zip = tmp_path / "test.bundle.zip"
    result = runner.invoke(
        app,
        [
            "bundle",
            "create",
            str(env_path),
            "--out",
            str(bundle_zip),
            "--include-public-key",
            str(pub_path),
        ],
    )
    assert result.exit_code == 0, result.stdout + (result.stderr or "")

    extract_dir = tmp_path / "ext"
    extract_dir.mkdir()
    with zipfile.ZipFile(bundle_zip) as zf:
        zf.extractall(extract_dir)

    proc = subprocess.run(
        [sys.executable, "verify.py", "--pubkey", "cosign.pub"],
        cwd=extract_dir,
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"stdout={proc.stdout}\nstderr={proc.stderr}"
    assert "OK" in proc.stdout
    assert "content_hash" in proc.stdout


def test_standalone_verify_script_detects_tamper(
    tmp_path: Path, dev_keypair: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mutating envelope.json must cause verify.py to FAIL with exit 1."""
    monkeypatch.chdir(tmp_path)
    env_path, pub_path = _signed_envelope_on_disk(tmp_path, dev_keypair)

    bundle_zip = tmp_path / "t.bundle.zip"
    runner.invoke(
        app,
        [
            "bundle",
            "create",
            str(env_path),
            "--out",
            str(bundle_zip),
            "--include-public-key",
            str(pub_path),
        ],
    )

    extract_dir = tmp_path / "tampered"
    extract_dir.mkdir()
    with zipfile.ZipFile(bundle_zip) as zf:
        zf.extractall(extract_dir)

    # Tamper: bump the seed in envelope.json.
    env_file = extract_dir / "envelope.json"
    data = json.loads(env_file.read_text(encoding="utf-8"))
    data["seed"] = int(data["seed"]) + 1
    env_file.write_text(json.dumps(data, sort_keys=True, indent=2), encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, "verify.py", "--pubkey", "cosign.pub"],
        cwd=extract_dir,
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 1
    combined = proc.stdout + proc.stderr
    assert "FAIL" in combined


def test_standalone_verify_script_keyless_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """For a sigstore-cosign envelope, verify.py defers to bench verify / cosign."""
    monkeypatch.chdir(tmp_path)
    env = make_envelope(model_id="m", metrics={"x": 1.0})
    # Manually attach a fake sigstore-cosign signature so we don't need a real OIDC flow.
    fake_signed = env.model_copy(
        update={
            "signature": Signature(
                method="sigstore-cosign",
                certificate="-----BEGIN CERTIFICATE-----\nfake\n-----END CERTIFICATE-----",
                rekor_log_index=42,
                bundle="ZmFrZQ==",
            )
        }
    )
    env_path = write_envelope_json(tmp_path / "envelope.json", fake_signed)

    bundle_zip = tmp_path / "k.bundle.zip"
    result = runner.invoke(app, ["bundle", "create", str(env_path), "--out", str(bundle_zip)])
    assert result.exit_code == 0

    extract_dir = tmp_path / "k"
    extract_dir.mkdir()
    with zipfile.ZipFile(bundle_zip) as zf:
        zf.extractall(extract_dir)

    proc = subprocess.run(
        [sys.executable, "verify.py"],
        cwd=extract_dir,
        check=False,
        capture_output=True,
        text=True,
    )
    # Keyless deferral: exit 2, with a hint about bench verify / cosign.
    assert proc.returncode == 2
    combined = proc.stdout + proc.stderr
    assert "Sigstore" in combined or "sigstore" in combined.lower()
