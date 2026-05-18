"""Adapters that convert raw Hugging Face rows into our local jsonl schema.

One function per ``adapter`` key referenced from
:data:`inferencebench.commands._fixtures_registry.FIXTURES`. Each adapter takes
an iterable of dict rows (the shape :func:`datasets.load_dataset` yields) and
yields canonical dicts ready to be written as one JSON object per line.

Output schemas are pinned by the consuming plugins:

- :func:`flores_pair` matches the ``llm-mt`` plugin's
  ``{"source","reference","domain"}`` fixture shape.
- :func:`humaneval` matches the ``code-generation`` plugin's HumanEval shape and
  refuses any row whose ``tests`` field imports a disallowed module the code
  runner refuses to execute.
- :func:`gsm8k` matches the ``llm-quality`` plugin's ``{"question","answer",
  "category"}`` shape, extracting the post-``####`` integer answer.
- :func:`truthfulqa_mc` matches the same ``llm-quality`` shape, picking the
  ``mc1_targets`` choice whose label is 1.
- :func:`msmarco_passage` matches the ``embeddings-retrieval`` plugin's
  ``{"query","relevant_doc_ids","corpus_size_hint"}`` query shape and caps
  output to the first 100 rows.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Iterator
from typing import Any

# Imports forbidden inside HumanEval ``tests`` strings — the bundled code runner
# refuses to execute these. Adapter silently drops such rows so we never write
# them into the cache.
_FORBIDDEN_TEST_IMPORTS: tuple[str, ...] = (
    "subprocess",
    "os.system",
    "socket",
    "urllib",
    "multiprocessing",
    "ctypes",
)

# Used by :func:`gsm8k` to extract the canonical answer after the ``####`` line.
_GSM8K_ANSWER_RE = re.compile(r"####\s*([^\s]+)")


def flores_pair(rows: Iterable[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    """Yield ``{"source","reference","domain"}`` rows from FLORES-200.

    A FLORES row has ``sentence_eng_Latn`` plus one ``sentence_<lang>_<script>``
    field for the target language. We pair the English source with whichever
    other ``sentence_*`` field is present and hardcode the domain to ``flores``.
    """
    for row in rows:
        source = row.get("sentence_eng_Latn")
        if not isinstance(source, str) or not source.strip():
            continue
        reference: str | None = None
        for key, value in row.items():
            if (
                key.startswith("sentence_")
                and key != "sentence_eng_Latn"
                and isinstance(value, str)
                and value.strip()
            ):
                reference = value
                break
        if reference is None:
            continue
        yield {
            "source": source,
            "reference": reference,
            "domain": "flores",
        }


def humaneval(rows: Iterable[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    """Yield HumanEval problems with disallowed-import rows filtered out.

    The Python code runner that grades these refuses to execute any test string
    that imports ``subprocess``, ``os.system``, ``socket``, ``urllib``,
    ``multiprocessing``, or ``ctypes``. Any such row is silently dropped here so
    those forbidden strings never even reach the cache file.
    """
    for row in rows:
        task_id = row.get("task_id")
        prompt = row.get("prompt")
        tests = row.get("test")
        canonical_solution = row.get("canonical_solution", "")
        entry_point = row.get("entry_point", "")
        # HF uses ``test`` (singular); accept ``tests`` too for adapter-level
        # input flexibility.
        if tests is None:
            tests = row.get("tests")
        if not isinstance(task_id, str) or not isinstance(prompt, str):
            continue
        if not isinstance(tests, str):
            continue
        if any(forbidden in tests for forbidden in _FORBIDDEN_TEST_IMPORTS):
            continue
        yield {
            "task_id": task_id,
            "prompt": prompt,
            "tests": tests,
            "canonical_solution": str(canonical_solution or ""),
            "entry_point": str(entry_point or ""),
        }


def gsm8k(rows: Iterable[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    """Yield ``{"question","answer","category"}`` rows from GSM8K.

    The HF ``answer`` field contains reasoning followed by ``#### <number>``;
    we extract just the number so deterministic scoring can do a clean match.
    """
    for row in rows:
        question = row.get("question")
        answer_blob = row.get("answer")
        if not isinstance(question, str) or not isinstance(answer_blob, str):
            continue
        match = _GSM8K_ANSWER_RE.search(answer_blob)
        if match is None:
            continue
        yield {
            "question": question,
            "answer": match.group(1).strip(),
            "category": "arithmetic",
        }


def truthfulqa_mc(rows: Iterable[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    """Yield ``{"question","answer","category"}`` rows from TruthfulQA-MC.

    ``mc1_targets`` is a dict with parallel ``choices`` and ``labels`` lists;
    the canonical correct answer is the choice whose label is 1.
    """
    for row in rows:
        question = row.get("question")
        targets = row.get("mc1_targets")
        if not isinstance(question, str) or not isinstance(targets, dict):
            continue
        choices = targets.get("choices")
        labels = targets.get("labels")
        if not isinstance(choices, list) or not isinstance(labels, list):
            continue
        if len(choices) != len(labels):
            continue
        canonical: str | None = None
        for choice, label in zip(choices, labels, strict=False):
            if int(label) == 1 and isinstance(choice, str):
                canonical = choice
                break
        if canonical is None:
            continue
        yield {
            "question": question,
            "answer": canonical,
            "category": "factual",
        }


def msmarco_passage(rows: Iterable[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    """Yield the first 100 MS MARCO passage queries in retrieval shape.

    HF MS-MARCO rows look like ``{"query": str, "passages": {"passage_text":
    [...], "is_selected": [0/1, ...], "url": [...]}}``. We surface the
    ``is_selected`` indices as opaque relevant-doc ids and add a corpus-size
    hint equal to the candidate passage count so the consuming plugin knows
    what to expect.
    """
    for idx, row in enumerate(rows):
        if idx >= 100:
            return
        query = row.get("query")
        passages = row.get("passages")
        if not isinstance(query, str) or not isinstance(passages, dict):
            continue
        is_selected = passages.get("is_selected")
        if not isinstance(is_selected, list):
            continue
        relevant_doc_ids = [str(i) for i, flag in enumerate(is_selected) if int(flag) == 1]
        yield {
            "query": query,
            "relevant_doc_ids": relevant_doc_ids,
            "corpus_size_hint": len(is_selected),
        }


ADAPTERS: dict[str, Callable[[Iterable[dict[str, Any]]], Iterator[dict[str, Any]]]] = {
    "flores_pair": flores_pair,
    "humaneval": humaneval,
    "gsm8k": gsm8k,
    "truthfulqa_mc": truthfulqa_mc,
    "msmarco_passage": msmarco_passage,
}
