"""Static validation tests for the repo-root Dockerfile + container helper.

These tests do NOT invoke ``docker`` — they only sanity-check that the
Dockerfile, ``.dockerignore``, and ``scripts/run_in_container.sh`` are
present and well-formed. Building the image happens in CI on a host that
has Docker installed.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_dockerfile_exists_at_repo_root() -> None:
    """``Dockerfile`` must live at the repo root for ``docker build .`` to work."""
    path = REPO_ROOT / "Dockerfile"
    assert path.is_file(), f"Dockerfile not found at {path}"


@pytest.mark.parametrize(
    "instruction",
    ["FROM", "WORKDIR", "COPY", "RUN", "ENTRYPOINT"],
)
def test_dockerfile_has_required_instruction(instruction: str) -> None:
    """Each required Dockerfile instruction must appear at least once.

    We grep at the start of a (logical) line to avoid matching the word
    inside a comment or a shell command.
    """
    body = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")
    pattern = re.compile(rf"^{instruction}\s+", re.MULTILINE)
    assert pattern.search(body), f"Dockerfile missing instruction: {instruction}"


def test_dockerfile_is_multi_stage() -> None:
    """Two ``FROM`` lines confirm the multi-stage build (builder + runtime)."""
    body = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")
    from_lines = re.findall(r"^FROM\s+", body, flags=re.MULTILINE)
    assert len(from_lines) >= 2, "Dockerfile should be multi-stage (>=2 FROM lines)"


def test_dockerignore_excludes_sensitive_paths() -> None:
    """``.dockerignore`` must exclude VCS, run artefacts, keys, and env files."""
    path = REPO_ROOT / ".dockerignore"
    assert path.is_file(), f".dockerignore not found at {path}"
    entries = {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    }
    for required in (".git", "validation-runs", "cosign.key", ".env"):
        assert required in entries, f".dockerignore missing entry: {required}"


def test_run_in_container_script_exists_and_is_executable() -> None:
    """Helper script must be present and executable so ``./scripts/run_in_container.sh`` works."""
    path = REPO_ROOT / "scripts" / "run_in_container.sh"
    assert path.is_file(), f"run_in_container.sh not found at {path}"
    assert os.access(path, os.X_OK), f"run_in_container.sh is not executable: {path}"
