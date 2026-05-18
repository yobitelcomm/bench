"""Bundled data files shipped inside the ``inferencebench`` CLI wheel.

Currently contains:

- ``plugin-registry.json`` — a build-time copy of
  ``tools/plugin-registry/registry.json``. Synced at release time; the
  source of truth lives at the repository top level. The test
  ``tests/test_plugin_registry_sync.py`` enforces content equality so the
  two files never drift.
"""

from __future__ import annotations
