"""Shared fixtures for the voice-transcription plugin tests.

The skeleton never makes outbound calls (audio is not decoded), so this
conftest exists mainly to anchor pytest's test discovery for the package
and to expose a ``dev_signing_ctx`` helper that mints a dev keypair on
demand. Future revisions that wire a real ASR engine can mock it here.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from inferencebench.envelope import generate_dev_keypair
from inferencebench_voice import EngineKind, RunContext


@pytest.fixture
def make_run_context(tmp_path: Path) -> Callable[..., RunContext]:
    """Return a factory that builds a signed-with-dev-key RunContext.

    Usage::

        def test_x(make_run_context):
            ctx = make_run_context()
            envelope = plugin.run(spec, ctx)
    """

    def _factory(
        *,
        engine_kind: EngineKind = EngineKind.OPENAI,
        model_id: str = "openai/whisper-mock",
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
