# inferencebench-quality

Quality (accuracy) plugin for the InferenceBench Suite.

Scores model answers against bundled fixture ground-truth using deterministic
exact-match, substring-match, or token-F1 strategies. LLM-as-judge is deferred
to a later revision — this is the contract-validation skeleton that proves the
plugin abstraction is not vLLM/perf-specific.

Suite ID: `llm.quality`

Bundled benchmarks:

- `llm.quality.factual-mini` — 10 short factual questions, substring-match scoring.
- `llm.quality.reasoning-mini` — 10 GSM8K-style single-number answers, exact-match scoring.
