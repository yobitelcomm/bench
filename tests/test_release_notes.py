"""Tests for scripts/release_notes.py."""

from __future__ import annotations

import importlib.util
import io
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "release_notes.py"


def _load_module() -> Any:
    spec = importlib.util.spec_from_file_location("release_notes", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["release_notes"] = module
    spec.loader.exec_module(module)
    return module


rn = _load_module()


def _capture_stdout(argv: list[str]) -> tuple[int, str, str]:
    """Run rn.main(argv), capturing stdout + stderr."""
    out_buf = io.StringIO()
    err_buf = io.StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = out_buf, err_buf
    try:
        code = rn.main(argv)
    finally:
        sys.stdout, sys.stderr = old_out, old_err
    return code, out_buf.getvalue(), err_buf.getvalue()


def test_release_notes_0_0_2_produces_non_empty_markdown():
    code, out, err = _capture_stdout(["--version", "0.0.2"])
    assert code == 0, err
    assert out.strip(), "expected non-empty markdown body"
    # Title carries the requested version.
    assert out.startswith("# v0.0.2"), out[:80]
    # Body length sanity — the changelog section is substantial.
    assert len(out) > 500, len(out)


def test_release_notes_0_0_2_contains_added_subsection():
    """Mirrors the Keep a Changelog convention — must surface an Added group."""
    code, out, _err = _capture_stdout(["--version", "0.0.2"])
    assert code == 0
    assert ("## Added" in out) or ("### Added" in out), out[:400]


def test_release_notes_unknown_version_exits_1():
    code, _out, err = _capture_stdout(["--version", "0.0.99"])
    assert code == 1
    assert "0.0.99" in err
    assert "not found" in err.lower() or "available" in err.lower()
