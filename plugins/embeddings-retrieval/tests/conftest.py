"""Shared fixtures for the embeddings-retrieval plugin tests.

The skeleton never embeds any text — rankings are produced by hashing —
so this conftest only exposes a ``make_run_context`` factory that mints a
dev keypair on demand.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from inferencebench.envelope import generate_dev_keypair
from inferencebench_embeddings import EngineKind, RunContext


@pytest.fixture
def make_run_context(tmp_path: Path) -> Callable[..., RunContext]:
    """Return a factory that builds a signed-with-dev-key RunContext."""

    def _factory(
        *,
        engine_kind: EngineKind = EngineKind.OPENAI,
        model_id: str = "openai/text-embedding-3-mock",
    ) -> RunContext:
        key_path = tmp_path / "cosign.key"
        generate_dev_keypair(key_path)
        return RunContext(
            model_id=model_id,
            model_revision="abc1234",
            engine_kind=engine_kind,
            output_dir=tmp_path / "out",
            extra={"signing_mode": "dev", "dev_key_path": str(key_path)},
        )

    return _factory
