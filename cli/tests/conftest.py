"""Pytest fixtures shared across the CLI test suite.

Helpers for building/signing envelopes live in ``cli/tests/_helpers.py`` so
they can be imported unambiguously alongside other workspace ``conftest.py``
modules. This file only provides fixtures + the ``COLUMNS=240`` shim that
keeps Rich tables wide enough for output-substring assertions.
"""

from __future__ import annotations

import os

# Force-set (not setdefault) — CI runners may have COLUMNS pre-set to a narrow
# value that wraps Rich help-text and breaks substring assertions on flag names.
os.environ["COLUMNS"] = "240"
# Disable ANSI escapes inside the CliRunner so substring assertions don't have
# to strip color codes. This only affects the test harness — production CLI
# output keeps its colour.
os.environ["NO_COLOR"] = "1"
os.environ["TERM"] = "dumb"

from pathlib import Path

import pytest
from rich.console import Console

from inferencebench.envelope import generate_dev_keypair


def _patch_cli_consoles() -> None:
    """Replace every CLI-module Console with a wide no-colour one.

    The cli modules instantiate ``Console()`` at module-import time which
    captures $COLUMNS at that single moment. By the time CliRunner sets its
    own ``env={"COLUMNS": "240"}`` the Console width is already pinned. We
    walk the imported submodules and swap the consoles to one that always
    renders wide + colourless, suitable for substring assertions.
    """
    wide = Console(width=240, no_color=True, force_terminal=False)
    wide_err = Console(width=240, no_color=True, force_terminal=False, stderr=True)
    import inferencebench.cli  # noqa: PLC0415  — deferred to make pytest import order deterministic

    for mod in list(sys.modules.values()):
        if (
            mod is not None
            and getattr(mod, "__name__", "").startswith("inferencebench.")
            and getattr(mod, "console", None) is not None
        ):
            if isinstance(mod.console, Console):
                mod.console = wide
            if isinstance(getattr(mod, "err_console", None), Console):
                mod.err_console = wide_err
    _ = inferencebench.cli  # keep import side-effect


import sys  # noqa: E402  — imported lazily for _patch_cli_consoles

_patch_cli_consoles()


@pytest.fixture
def dev_keypair(tmp_path: Path) -> tuple[Path, Path]:
    """Fresh ed25519 dev keypair scoped per-test."""
    return generate_dev_keypair(tmp_path / "cosign.key")
