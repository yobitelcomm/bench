# inferencebench-mt

Machine-translation plugin for the InferenceBench Suite.

Scores model translations against bundled reference fixtures using chrF (character
n-gram F-score), token-level BLEU, or exact match. Mirrors the contract of the
other plugins (`list_benchmarks` / `get_benchmark` / `validate` / `run`) and
emits the canonical signed envelope.

Two bundled benchmarks ship out of the box:

- `llm.mt.flores-200-mini-en-fr` — FLORES-200-style English to French, chrF.
- `llm.mt.flores-200-mini-en-de` — FLORES-200-style English to German, chrF.

The fixtures are tiny (eight rows each, mixed across greeting / news / technical
/ conversational domains) — intended for skeleton verification, not headline
numbers.
