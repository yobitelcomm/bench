"""Registry of known public dataset fixtures.

The :data:`FIXTURES` dict maps stable user-facing keys (used as ``fixtures://<key>``
URIs and in ``bench fixtures fetch <key>``) to :class:`FixtureEntry` records that
describe how to download the corresponding Hugging Face dataset and which adapter
converts the raw rows into our local jsonl schema.

This module is intentionally pure data — no network access, no I/O — so it can be
imported by both the CLI command and any plugin that needs to introspect fixture
metadata.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FixtureEntry:
    """One downloadable fixture.

    Attributes:
        hf_dataset: The Hugging Face dataset repo id passed to
            :func:`datasets.load_dataset` (e.g. ``"facebook/flores"``).
        hf_config: Optional config / subset name (e.g. ``"eng_Latn-fra_Latn"``).
        split: Which split to pull (``"train"``, ``"validation"``, ``"test"``, ...).
        license: SPDX-style license identifier — surfaced in the fetch summary.
        size_estimate_mb: Rough on-disk size after conversion. Surfaced to users
            before they kick off a download.
        description: One-line human description.
        adapter: Key into :data:`inferencebench.commands._fixtures_adapters.ADAPTERS`
            naming the row converter for this dataset.
    """

    hf_dataset: str
    hf_config: str | None
    split: str
    license: str
    size_estimate_mb: int
    description: str
    adapter: str


FIXTURES: dict[str, FixtureEntry] = {
    "flores-200-eng-fra": FixtureEntry(
        hf_dataset="facebook/flores",
        hf_config="eng_Latn-fra_Latn",
        split="dev",
        license="CC-BY-SA-4.0",
        size_estimate_mb=2,
        description="FLORES-200 English to French parallel sentences (dev split).",
        adapter="flores_pair",
    ),
    "flores-200-eng-deu": FixtureEntry(
        hf_dataset="facebook/flores",
        hf_config="eng_Latn-deu_Latn",
        split="dev",
        license="CC-BY-SA-4.0",
        size_estimate_mb=2,
        description="FLORES-200 English to German parallel sentences (dev split).",
        adapter="flores_pair",
    ),
    "flores-200-eng-spa": FixtureEntry(
        hf_dataset="facebook/flores",
        hf_config="eng_Latn-spa_Latn",
        split="dev",
        license="CC-BY-SA-4.0",
        size_estimate_mb=2,
        description="FLORES-200 English to Spanish parallel sentences (dev split).",
        adapter="flores_pair",
    ),
    "flores-200-eng-jpn": FixtureEntry(
        hf_dataset="facebook/flores",
        hf_config="eng_Latn-jpn_Jpan",
        split="dev",
        license="CC-BY-SA-4.0",
        size_estimate_mb=2,
        description="FLORES-200 English to Japanese parallel sentences (dev split).",
        adapter="flores_pair",
    ),
    "humaneval": FixtureEntry(
        hf_dataset="openai_humaneval",
        hf_config=None,
        split="test",
        license="MIT",
        size_estimate_mb=1,
        description="OpenAI HumanEval 164 hand-written code-generation problems.",
        adapter="humaneval",
    ),
    "gsm8k": FixtureEntry(
        hf_dataset="gsm8k",
        hf_config="main",
        split="test",
        license="MIT",
        size_estimate_mb=4,
        description="GSM8K grade-school math word problems (test split).",
        adapter="gsm8k",
    ),
    "truthfulqa-mc": FixtureEntry(
        hf_dataset="truthful_qa",
        hf_config="multiple_choice",
        split="validation",
        license="Apache-2.0",
        size_estimate_mb=1,
        description="TruthfulQA multiple-choice validation set.",
        adapter="truthfulqa_mc",
    ),
    "ms-marco-passage-mini": FixtureEntry(
        hf_dataset="ms_marco",
        hf_config="v2.1",
        split="validation",
        license="MIT",
        size_estimate_mb=15,
        description="MS MARCO passage ranking validation queries (first 100).",
        adapter="msmarco_passage",
    ),
}
