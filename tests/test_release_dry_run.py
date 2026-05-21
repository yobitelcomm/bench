"""Tests for scripts/release_dry_run.py."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
import zipfile
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "release_dry_run.py"


def _load_module() -> Any:
    """Import the script as a module without polluting sys.path globally."""
    spec = importlib.util.spec_from_file_location("release_dry_run", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["release_dry_run"] = module
    spec.loader.exec_module(module)
    return module


rdr = _load_module()


EXPECTED_PACKAGE_NAMES = {
    "cli",
    "envelope",
    "harness",
    "llm-inference",
    "llm-quality",
    "voice-transcription",
    "embeddings-retrieval",
    "hf-publisher",
    "leaderboard",
}


def test_discover_packages_finds_at_least_nine():
    pkgs = rdr.discover_packages()
    assert len(pkgs) >= 9, f"expected >=9 packages, got {len(pkgs)}: {pkgs}"
    leaf_names = {p.name for p in pkgs}
    missing = EXPECTED_PACKAGE_NAMES - leaf_names
    assert not missing, f"missing expected packages: {missing}"


def test_discover_packages_excludes_root_pyproject():
    pkgs = rdr.discover_packages()
    resolved_root = REPO_ROOT.resolve()
    assert resolved_root not in pkgs


def _make_fake_workspace(root: Path) -> list[Path]:
    """Build a tmp workspace mirroring the repo's globbing patterns."""
    layout = [
        root / "cli",
        root / "envelope",
        root / "plugins" / "alpha",
        root / "plugins" / "beta",
        root / "integrations" / "gamma",
        root / "tools" / "delta",
    ]
    written: list[Path] = []
    for d in layout:
        d.mkdir(parents=True, exist_ok=True)
        py = d / "pyproject.toml"
        py.write_text(
            f'[project]\nname = "fake-{d.name}"\nversion = "0.0.0"\ndescription = "x"\n',
            encoding="utf-8",
        )
        written.append(py)
    # A non-package pyproject that looks like the workspace root — must be
    # skipped.
    (root / "pyproject.toml").write_text(
        '[project]\nname = "root"\nversion = "9.9.9"\n',
        encoding="utf-8",
    )
    return written


def test_update_versions_dry_run_returns_paths_without_writing(tmp_path):
    written = _make_fake_workspace(tmp_path)
    before = {p: p.read_text(encoding="utf-8") for p in written}

    changed = rdr.update_versions("0.0.2", dry_run=True, repo_root=tmp_path)

    assert set(changed) == set(written)
    for p, original in before.items():
        assert p.read_text(encoding="utf-8") == original, "dry_run must not write"


def test_update_versions_rewrites_in_place(tmp_path):
    written = _make_fake_workspace(tmp_path)

    changed = rdr.update_versions("0.0.2", dry_run=False, repo_root=tmp_path)
    assert set(changed) == set(written)

    for p in written:
        text = p.read_text(encoding="utf-8")
        assert 'version = "0.0.2"' in text
        assert 'version = "0.0.0"' not in text

    # Root pyproject (not a workspace member under our globs) is untouched.
    root_text = (tmp_path / "pyproject.toml").read_text(encoding="utf-8")
    assert 'version = "9.9.9"' in root_text


def test_update_versions_idempotent(tmp_path):
    _make_fake_workspace(tmp_path)
    rdr.update_versions("0.0.2", dry_run=False, repo_root=tmp_path)
    second = rdr.update_versions("0.0.2", dry_run=False, repo_root=tmp_path)
    # Second run finds nothing to change.
    assert second == []


def _build_synthetic_wheel(dest_dir: Path, *, with_metadata_fields: bool = True) -> Path:
    """Construct a minimal valid wheel zip on disk, return its path."""
    name = "fakepkg"
    version = "0.0.1"
    dist_info = f"{name}-{version}.dist-info"
    wheel_path = dest_dir / f"{name}-{version}-py3-none-any.whl"

    if with_metadata_fields:
        metadata = (
            "Metadata-Version: 2.1\n"
            f"Name: {name}\n"
            f"Version: {version}\n"
            "Summary: a fake package\n"
            "License: Apache-2.0\n"
            "Requires-Python: >=3.12\n"
        )
    else:
        metadata = f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n"
    wheel_file = "Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n"
    pkg_init = "# fake\n"

    files = {
        f"{name}/__init__.py": pkg_init.encode(),
        f"{dist_info}/METADATA": metadata.encode(),
        f"{dist_info}/WHEEL": wheel_file.encode(),
    }

    record_lines = []
    for fname, data in files.items():
        digest = hashlib.sha256(data).digest()
        b64 = __import__("base64").urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
        record_lines.append(f"{fname},sha256={b64},{len(data)}")
    record_lines.append(f"{dist_info}/RECORD,,")
    record = ("\n".join(record_lines) + "\n").encode()
    files[f"{dist_info}/RECORD"] = record

    with zipfile.ZipFile(wheel_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for fname, data in files.items():
            zf.writestr(fname, data)
    return wheel_path


def test_validate_wheel_passes_on_well_formed_wheel(tmp_path):
    wheel = _build_synthetic_wheel(tmp_path, with_metadata_fields=True)
    issues = rdr.validate_wheel(wheel)
    assert issues == [], f"expected no issues, got {issues}"


def test_validate_wheel_flags_missing_metadata(tmp_path):
    wheel = _build_synthetic_wheel(tmp_path, with_metadata_fields=False)
    issues = rdr.validate_wheel(wheel)
    # Summary, License, Requires-Python all missing in the stripped wheel.
    joined = " | ".join(issues)
    assert "Summary" in joined
    assert "License" in joined
    assert "Requires-Python" in joined


def test_parse_args_rejects_bump_with_check_only():
    with pytest.raises(SystemExit):
        rdr.parse_args(["--bump", "0.0.2", "--check-only"])


def test_parse_args_defaults():
    args = rdr.parse_args([])
    assert args.bump is None
    assert args.check_only is False
    assert args.clean is False
    assert args.verbose is False
