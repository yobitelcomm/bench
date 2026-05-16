"""Tests for the llm-inference dataset loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from inferencebench_llm.datasets import compute_dataset_hash, load_prompts
from inferencebench_llm.schemas import DatasetConfig, DatasetSamplingConfig


def _spec(uri: str, n: int = 5, seed: int = 42) -> DatasetConfig:
    return DatasetConfig(
        id="test",
        uri=uri,
        hash="sha256:" + "0" * 64,
        sampling=DatasetSamplingConfig(n=n, seed=seed),
    )


# --------------------------------------------------------------------------- #
# Schema fallback                                                             #
# --------------------------------------------------------------------------- #
def test_builtin_uri_returns_fallback_prompts() -> None:
    prompts = load_prompts(_spec("builtin://", n=5))
    assert len(prompts) == 5
    assert all(isinstance(p, str) and p for p in prompts)


def test_builtin_with_larger_n_repeats() -> None:
    """If n exceeds the builtin pool, the loader repeats."""
    prompts = load_prompts(_spec("builtin://", n=25))
    assert len(prompts) == 25


def test_builtin_with_smaller_n_truncates() -> None:
    prompts = load_prompts(_spec("builtin://", n=3))
    assert len(prompts) == 3


# --------------------------------------------------------------------------- #
# file:// loader                                                              #
# --------------------------------------------------------------------------- #
def test_file_loader_reads_jsonl(tmp_path: Path) -> None:
    p = tmp_path / "prompts.jsonl"
    p.write_text(
        '{"prompt": "Test 1"}\n'
        '{"prompt": "Test 2"}\n'
        '"plain string"\n'
        '{"not_a_prompt": "ignored"}\n'
        "not-json\n"
    )
    prompts = load_prompts(_spec(f"file://{p}", n=10))
    # Loader yields 3 valid prompts; n=10 means it repeats to fill
    assert len(prompts) == 10
    assert "Test 1" in prompts
    assert "Test 2" in prompts
    assert "plain string" in prompts


def test_file_loader_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_prompts(_spec("file:///nonexistent.jsonl", n=3))


# --------------------------------------------------------------------------- #
# hf:// loader (offline-friendly)                                             #
# --------------------------------------------------------------------------- #
def test_hf_loader_falls_back_when_offline() -> None:
    """Unreachable HF URI yields fallback prompts, doesn't raise."""
    prompts = load_prompts(_spec("hf://nonexistent-org/nonexistent-dataset", n=5))
    assert len(prompts) == 5


# --------------------------------------------------------------------------- #
# Unknown scheme                                                              #
# --------------------------------------------------------------------------- #
def test_unknown_scheme_raises() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        load_prompts(_spec("ftp://example.com/data.txt", n=5))


# --------------------------------------------------------------------------- #
# compute_dataset_hash                                                        #
# --------------------------------------------------------------------------- #
def test_dataset_hash_is_deterministic() -> None:
    prompts = ["alpha", "beta", "gamma"]
    h1 = compute_dataset_hash(prompts)
    h2 = compute_dataset_hash(prompts)
    assert h1 == h2
    assert len(h1) == 64
    assert all(c in "0123456789abcdef" for c in h1)


def test_dataset_hash_changes_with_content() -> None:
    a = compute_dataset_hash(["alpha", "beta"])
    b = compute_dataset_hash(["alpha", "beta", "gamma"])
    c = compute_dataset_hash(["gamma", "beta", "alpha"])  # different order
    assert a != b
    assert b != c
