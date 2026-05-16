"""Pytest fixtures shared across the CLI test suite.

Helpers for building/signing envelopes live in ``cli/tests/_helpers.py`` so
they can be imported unambiguously alongside other workspace ``conftest.py``
modules. This file only provides fixtures + the ``COLUMNS=240`` shim that
keeps Rich tables wide enough for output-substring assertions.
"""

from __future__ import annotations

import os

os.environ.setdefault("COLUMNS", "240")

from pathlib import Path

import pytest

from inferencebench.envelope import generate_dev_keypair


@pytest.fixture
def dev_keypair(tmp_path: Path) -> tuple[Path, Path]:
    """Fresh ed25519 dev keypair scoped per-test."""
    return generate_dev_keypair(tmp_path / "cosign.key")
