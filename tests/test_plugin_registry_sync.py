"""Drift-guard: the bundled registry stays in sync with the source-of-truth.

``tools/plugin-registry/registry.json`` is the canonical registry; the
copy at ``cli/src/inferencebench/data/plugin-registry.json`` is a build
artifact synced at release time. This test asserts the two files have
identical parsed content so they cannot silently diverge.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE = REPO_ROOT / "tools" / "plugin-registry" / "registry.json"
BUNDLED = REPO_ROOT / "cli" / "src" / "inferencebench" / "data" / "plugin-registry.json"


def test_source_and_bundled_registry_have_identical_content() -> None:
    """Source-of-truth and bundled copy must parse to identical JSON."""
    assert SOURCE.exists(), f"source registry missing: {SOURCE}"
    assert BUNDLED.exists(), f"bundled registry missing: {BUNDLED}"
    src = json.loads(SOURCE.read_text(encoding="utf-8"))
    bundled = json.loads(BUNDLED.read_text(encoding="utf-8"))
    assert src == bundled, (
        "tools/plugin-registry/registry.json and "
        "cli/src/inferencebench/data/plugin-registry.json have drifted. "
        "Re-run the sync step: `cp tools/plugin-registry/registry.json "
        "cli/src/inferencebench/data/plugin-registry.json`."
    )
