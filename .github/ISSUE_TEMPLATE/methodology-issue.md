---
name: Methodology concern
about: A benchmark's methodology may be biased, contaminated, or game-able
title: "[methodology] "
labels: methodology
---

## Which benchmark

`<suite_id>` (e.g. `llm.inference.sharegpt-v3`, `agent.coding:swe-bench-verified`)

## What's the concern

Pick one or more:

- [ ] Vendor bias (favors a specific hardware / engine / model family)
- [ ] Data contamination (test set leaked into training corpora)
- [ ] Reward-hacking surface (easy to game without actually solving the task)
- [ ] Reproducibility (running same workload twice produces different envelopes)
- [ ] Judge bias (LLM-as-judge has positional / style preferences)
- [ ] Cost / energy reporting inaccuracy
- [ ] Other:

## Evidence

Be specific. Bullet points, links to envelopes, reproduction scripts.

## Suggested remediation

What would fix it?

## Reference

- Methodology page: `docs/plugins/<modality>.md`
- Methodology review: `plugins/<modality>/docs/methodology-review.md`

---

*See [agents/benchmark-validator.md](../../agents/benchmark-validator.md) for the validator's review checklist. We treat methodology issues as P1.*
