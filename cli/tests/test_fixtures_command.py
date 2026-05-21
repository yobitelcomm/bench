"""Tests for ``bench fixtures`` — dataset fixture fetcher + registry.

Verifies the four documented subcommands (``list``, ``fetch``, ``path``,
``clear``) plus the per-adapter conversion logic. The HF download path is
exercised via a mock of :func:`datasets.load_dataset` — no network access in
CI. The cache root is redirected via ``BENCH_FIXTURES_ROOT`` so tests never
touch the real ``~/.cache``.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest
from typer.testing import CliRunner

from inferencebench.cli import app
from inferencebench.commands._fixtures_adapters import (
    flores_pair,
    gsm8k,
    humaneval,
    msmarco_passage,
    truthfulqa_mc,
)
from inferencebench.commands._fixtures_registry import FIXTURES

runner = CliRunner(env={"COLUMNS": "240"})


def _install_mock_datasets(monkeypatch: pytest.MonkeyPatch, rows: list[dict[str, object]]) -> None:
    """Install a stub ``datasets`` module whose ``load_dataset`` returns ``rows``."""
    module = types.ModuleType("datasets")

    def load_dataset(*_args: object, **_kwargs: object) -> list[dict[str, object]]:
        return list(rows)

    module.load_dataset = load_dataset  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "datasets", module)


# --------------------------------------------------------------------------- #
# list                                                                        #
# --------------------------------------------------------------------------- #
def test_fixtures_list_shows_known_keys(tmp_path: Path) -> None:
    """``bench fixtures list`` exits 0 and prints every fixture key."""
    cache = tmp_path / "fx"
    result = runner.invoke(
        app,
        ["fixtures", "list"],
        env={"BENCH_FIXTURES_ROOT": str(cache), "COLUMNS": "240"},
    )
    assert result.exit_code == 0, result.stdout + (result.stderr or "")
    # At least 7 keys per the registry.
    assert len(FIXTURES) >= 7
    for key in FIXTURES:
        assert key in result.stdout, f"missing {key} in:\n{result.stdout}"


# --------------------------------------------------------------------------- #
# path                                                                        #
# --------------------------------------------------------------------------- #
def test_fixtures_path_prints_cache_root(tmp_path: Path) -> None:
    """``fixtures path`` prints a single line equal to BENCH_FIXTURES_ROOT."""
    cache = tmp_path / "fx" / "cache"
    result = runner.invoke(
        app,
        ["fixtures", "path"],
        env={"BENCH_FIXTURES_ROOT": str(cache), "COLUMNS": "240"},
    )
    assert result.exit_code == 0, result.stdout + (result.stderr or "")
    lines = [ln for ln in result.stdout.splitlines() if ln.strip()]
    assert len(lines) == 1
    assert str(cache) == lines[0].strip()


# --------------------------------------------------------------------------- #
# fetch                                                                       #
# --------------------------------------------------------------------------- #
def test_fixtures_fetch_writes_jsonl_and_shows_cached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A mocked HumanEval fetch lands rows on disk; subsequent list marks cached."""
    cache = tmp_path / "fx"
    rows: list[dict[str, object]] = [
        {
            "task_id": "HumanEval/0",
            "prompt": 'def add(a, b):\n    """add a and b"""\n',
            "test": "def check(c): assert c(1,2) == 3",
            "canonical_solution": "    return a + b",
            "entry_point": "add",
        },
        {
            "task_id": "HumanEval/1",
            "prompt": "def sub(a, b):\n    return a - b",
            "test": "def check(c): assert c(2,1) == 1",
            "canonical_solution": "",
            "entry_point": "sub",
        },
    ]
    _install_mock_datasets(monkeypatch, rows)
    monkeypatch.setenv("BENCH_FIXTURES_ROOT", str(cache))

    result = runner.invoke(
        app,
        ["fixtures", "fetch", "humaneval"],
        env={"BENCH_FIXTURES_ROOT": str(cache), "COLUMNS": "240"},
    )
    assert result.exit_code == 0, result.stdout + (result.stderr or "")

    out_path = cache / "humaneval.jsonl"
    assert out_path.exists()
    written = [json.loads(line) for line in out_path.read_text().splitlines() if line.strip()]
    assert len(written) == 2
    assert {row["task_id"] for row in written} == {"HumanEval/0", "HumanEval/1"}
    assert "wrote 2 rows" in result.stdout

    # Subsequent list now marks the fixture cached.
    list_result = runner.invoke(
        app,
        ["fixtures", "list"],
        env={"BENCH_FIXTURES_ROOT": str(cache), "COLUMNS": "240"},
    )
    assert list_result.exit_code == 0
    assert "yes" in list_result.stdout  # cached column flips to 'yes' for humaneval


def test_fixtures_fetch_bad_key_exits_2(tmp_path: Path) -> None:
    """Unknown fixture key → exit 2 with a helpful error."""
    cache = tmp_path / "fx"
    result = runner.invoke(
        app,
        ["fixtures", "fetch", "definitely-not-a-fixture"],
        env={"BENCH_FIXTURES_ROOT": str(cache), "COLUMNS": "240"},
    )
    assert result.exit_code == 2
    combined = result.stdout + (result.stderr or "")
    assert "Unknown fixture key" in combined


def test_fixtures_fetch_no_datasets_package_exits_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If ``datasets`` is missing, fetch exits 2 and points the user to pip install."""
    cache = tmp_path / "fx"
    # Make ``import datasets`` fail by ensuring it's absent and finder rejects it.
    monkeypatch.setitem(sys.modules, "datasets", None)  # type: ignore[arg-type]
    result = runner.invoke(
        app,
        ["fixtures", "fetch", "humaneval"],
        env={"BENCH_FIXTURES_ROOT": str(cache), "COLUMNS": "240"},
    )
    assert result.exit_code == 2
    combined = result.stdout + (result.stderr or "")
    assert "pip install datasets" in combined


# --------------------------------------------------------------------------- #
# clear                                                                       #
# --------------------------------------------------------------------------- #
def test_fixtures_clear_specific_key_removes_only_that(tmp_path: Path) -> None:
    """``--key K --yes`` removes only K; other keys remain in the cache."""
    cache = tmp_path / "fx"
    cache.mkdir(parents=True)
    (cache / "humaneval.jsonl").write_text('{"task_id":"a","prompt":"x"}\n')
    (cache / "gsm8k.jsonl").write_text('{"question":"q","answer":"1"}\n')

    result = runner.invoke(
        app,
        ["fixtures", "clear", "--key", "humaneval", "--yes"],
        env={"BENCH_FIXTURES_ROOT": str(cache), "COLUMNS": "240"},
    )
    assert result.exit_code == 0, result.stdout + (result.stderr or "")
    assert not (cache / "humaneval.jsonl").exists()
    assert (cache / "gsm8k.jsonl").exists()


# --------------------------------------------------------------------------- #
# Adapters — schema unit tests                                                #
# --------------------------------------------------------------------------- #
def test_flores_pair_adapter_yields_documented_shape() -> None:
    rows = [
        {
            "id": 1,
            "sentence_eng_Latn": "Hello world.",
            "sentence_fra_Latn": "Bonjour le monde.",
        },
        {
            "id": 2,
            "sentence_eng_Latn": "",
            "sentence_fra_Latn": "should be dropped",
        },
    ]
    converted = list(flores_pair(rows))
    assert len(converted) == 1
    assert converted[0] == {
        "source": "Hello world.",
        "reference": "Bonjour le monde.",
        "domain": "flores",
    }


def test_humaneval_adapter_yields_documented_shape() -> None:
    rows = [
        {
            "task_id": "HumanEval/0",
            "prompt": "def f(x): pass",
            "test": "def check(c): pass",
            "canonical_solution": "    return x",
            "entry_point": "f",
        }
    ]
    converted = list(humaneval(rows))
    assert converted == [
        {
            "task_id": "HumanEval/0",
            "prompt": "def f(x): pass",
            "tests": "def check(c): pass",
            "canonical_solution": "    return x",
            "entry_point": "f",
        }
    ]


def test_humaneval_adapter_refuses_forbidden_imports() -> None:
    """Rows that import subprocess (or other forbidden modules) are dropped."""
    rows = [
        {
            "task_id": "HumanEval/safe",
            "prompt": "def f(): pass",
            "test": "def check(c): pass",
            "canonical_solution": "",
            "entry_point": "f",
        },
        {
            "task_id": "HumanEval/danger",
            "prompt": "def g(): pass",
            "test": "import subprocess\nsubprocess.run(['rm','-rf','/'])",
            "canonical_solution": "",
            "entry_point": "g",
        },
        {
            "task_id": "HumanEval/danger2",
            "prompt": "def h(): pass",
            "test": "import socket\nsocket.create_connection(('evil', 80))",
            "canonical_solution": "",
            "entry_point": "h",
        },
    ]
    converted = list(humaneval(rows))
    assert len(converted) == 1
    assert converted[0]["task_id"] == "HumanEval/safe"


def test_gsm8k_adapter_extracts_post_hash_answer() -> None:
    rows = [
        {
            "question": "If Alice has 3 apples and Bob has 5, how many total?",
            "answer": "Alice has 3 and Bob has 5; total is 3+5=8.\n#### 8",
        },
        {
            "question": "no marker",
            "answer": "nothing here",  # missing #### — skipped
        },
    ]
    converted = list(gsm8k(rows))
    assert len(converted) == 1
    assert converted[0] == {
        "question": "If Alice has 3 apples and Bob has 5, how many total?",
        "answer": "8",
        "category": "arithmetic",
    }


def test_truthfulqa_mc_adapter_picks_label_one_choice() -> None:
    rows = [
        {
            "question": "What is the capital of France?",
            "mc1_targets": {
                "choices": ["Berlin", "Paris", "Rome"],
                "labels": [0, 1, 0],
            },
        },
        {
            "question": "malformed",
            "mc1_targets": {"choices": ["x"], "labels": [0, 0]},  # length mismatch
        },
    ]
    converted = list(truthfulqa_mc(rows))
    assert converted == [
        {"question": "What is the capital of France?", "answer": "Paris", "category": "factual"}
    ]


def test_msmarco_passage_adapter_caps_at_100_and_uses_is_selected() -> None:
    rows = [
        {
            "query": f"query {i}",
            "passages": {
                "passage_text": [f"p{i}.0", f"p{i}.1", f"p{i}.2"],
                "is_selected": [0, 1, 0] if i % 2 == 0 else [1, 0, 0],
                "url": ["u0", "u1", "u2"],
            },
        }
        for i in range(120)
    ]
    converted = list(msmarco_passage(rows))
    assert len(converted) == 100
    first = converted[0]
    assert first["query"] == "query 0"
    assert first["relevant_doc_ids"] == ["1"]
    assert first["corpus_size_hint"] == 3
