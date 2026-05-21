"""Tests for the ``fixtures://`` URI scheme in the llm-inference loader.

The scheme resolves to the local cache ``bench fixtures fetch`` writes — we
redirect that cache via ``BENCH_FIXTURES_ROOT`` so the test owns its data and
never touches the user's real cache.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from inferencebench_llm.datasets import load_prompts
from inferencebench_llm.schemas import DatasetConfig, DatasetSamplingConfig


def _spec(uri: str, n: int = 3, seed: int = 42) -> DatasetConfig:
    return DatasetConfig(
        id="fixtures-test",
        uri=uri,
        hash="sha256:" + "0" * 64,
        sampling=DatasetSamplingConfig(n=n, seed=seed),
    )


def test_fixtures_uri_loads_cached_prompts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A cached fixtures jsonl resolves through the ``fixtures://`` scheme."""
    cache = tmp_path / "fx"
    cache.mkdir()
    (cache / "flores-200-eng-fra.jsonl").write_text(
        '{"source": "Hello.", "reference": "Bonjour.", "domain": "flores"}\n'
        '{"source": "Goodbye.", "reference": "Au revoir.", "domain": "flores"}\n'
        '{"source": "Thanks.", "reference": "Merci.", "domain": "flores"}\n'
    )
    monkeypatch.setenv("BENCH_FIXTURES_ROOT", str(cache))

    prompts = load_prompts(_spec("fixtures://flores-200-eng-fra", n=3))
    assert len(prompts) == 3
    # ``source`` is the chosen string field for flores rows.
    assert "Hello." in prompts
    assert "Goodbye." in prompts
    assert "Thanks." in prompts


def test_fixtures_uri_missing_cache_raises_with_hint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Missing cache → clear error pointing the user at ``bench fixtures fetch``."""
    cache = tmp_path / "fx"
    cache.mkdir()
    monkeypatch.setenv("BENCH_FIXTURES_ROOT", str(cache))

    with pytest.raises(FileNotFoundError, match="bench fixtures fetch"):
        load_prompts(_spec("fixtures://humaneval", n=3))
